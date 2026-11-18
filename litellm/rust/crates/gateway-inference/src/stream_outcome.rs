//! How a served stream went, told to the router as soon as it is known. A streamed attempt is
//! served deferred: the router counted its request and hears the rest here, a success with the
//! tokens the stream counted once it ends, or a failure the moment the stream breaks or carries
//! the provider's error. A stream the client stops reading reports neither.
//!
//! A provider's error event is the failure the same error is before a stream opens, with the
//! same status, so the cooldown rules read both alike. The Python proxy differs: it records a
//! stream that goes out with an error event as served, unless another attempt took it over.

use std::{future::Future, sync::Arc};

use bytes::Bytes;
use futures_util::{
    StreamExt, TryStreamExt,
    stream::{self, BoxStream},
};
use litellm_framer::sse::{self, SseBlock};
use litellm_host::{call::CallOutput, protocol::Protocol};
use litellm_inference::Error as InferenceError;
use litellm_llms_types::formats::{
    messages::streaming::{MessagesStreamEvent, MessagesStreamUsage},
    responses::streaming::ResponsesEventType,
};
use litellm_router_core::{
    AttemptError, Deployment, FailureKind, FailureReport, Router, RouterCache, SuccessReport,
};
use serde::Deserialize;
use serde_json::Value;

use crate::{Gateway, error::anthropic_error_status, routing};

/// The status of a Responses error event, which names none.
const RESPONSES_ERROR_STATUS: u16 = 500;

/// What a hosted call ran to, a stream of it as the blocks it was read in.
pub(crate) type Streamed<P> = CallOutput<
    <P as Protocol>::Response,
    <P as Protocol>::StreamHead,
    SseBlock,
    litellm_host_http::Error<<P as Protocol>::Error>,
>;

/// `streamed` as [`litellm_host_http::render`] takes it, a stream of it as the bytes of its
/// blocks.
pub(crate) fn bytes<R, H, E: 'static>(
    streamed: CallOutput<R, H, SseBlock, E>,
) -> CallOutput<R, H, Bytes, E> {
    match streamed {
        CallOutput::Complete(response) => CallOutput::Complete(response),
        CallOutput::Stream { head, chunks } => CallOutput::Stream {
            head,
            chunks: chunks.map_ok(|block| block.bytes).boxed(),
        },
    }
}

/// The tokens an event's `usage` counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize)]
struct Counts {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

impl Counts {
    fn over(self, earlier: Self) -> Self {
        Self {
            input_tokens: self.input_tokens.or(earlier.input_tokens),
            output_tokens: self.output_tokens.or(earlier.output_tokens),
            total_tokens: self.total_tokens.or(earlier.total_tokens),
        }
    }

    /// `total_tokens`, else the input plus the output, as [`routing::total_tokens`] counts a
    /// complete response.
    fn total(self) -> u64 {
        self.total_tokens
            .unwrap_or_else(|| self.input_tokens.unwrap_or(0) + self.output_tokens.unwrap_or(0))
    }
}

impl From<MessagesStreamUsage> for Counts {
    fn from(usage: MessagesStreamUsage) -> Self {
        Self {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens: None,
        }
    }
}

/// What a stream's own events said so far.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Told {
    /// The tokens the events counted, a later event's count over an earlier one's.
    counts: Counts,
    /// The provider's error event: the status the same error has before a stream, and the event
    /// as it was sent.
    error: Option<(u16, String)>,
}

impl Told {
    fn counting(self, usage: Option<impl Into<Counts>>) -> Self {
        let Some(usage) = usage else {
            return self;
        };
        Self {
            counts: usage.into().over(self.counts),
            ..self
        }
    }

    fn with_error(self, status: u16, sent: &str) -> Self {
        Self {
            error: Some((status, sent.to_owned())),
            ..self
        }
    }
}

/// Reads the data of one event of a protocol's stream into what the stream said so far.
pub(crate) type Read = fn(Told, &str) -> Told;

/// An Anthropic Messages event: `message_start` counts the input, `message_delta` the output.
pub(crate) fn messages_event(told: Told, data: &str) -> Told {
    match serde_json::from_str(data) {
        Ok(MessagesStreamEvent::MessageStart { message }) => told.counting(message.usage),
        Ok(
            MessagesStreamEvent::MessageDelta { usage, .. }
            | MessagesStreamEvent::MessageStop { usage, .. },
        ) => told.counting(usage),
        Ok(MessagesStreamEvent::Error { error }) => {
            told.with_error(anthropic_error_status(&error.error_type), data)
        }
        Ok(
            MessagesStreamEvent::ContentBlockStart { .. }
            | MessagesStreamEvent::ContentBlockDelta { .. }
            | MessagesStreamEvent::ContentBlockStop { .. }
            | MessagesStreamEvent::Ping
            | MessagesStreamEvent::Other(_),
        )
        | Err(_) => told,
    }
}

/// The part of a Responses event that carries its counts.
#[derive(Default, Deserialize)]
struct Counted {
    usage: Option<Counts>,
}

/// An OpenAI Responses event: the closing event carries the response and its usage.
#[derive(Deserialize)]
struct ResponsesEvent {
    #[serde(rename = "type")]
    kind: ResponsesEventType,
    response: Option<Counted>,
}

pub(crate) fn responses_event(told: Told, data: &str) -> Told {
    let Ok(event) = serde_json::from_str::<ResponsesEvent>(data) else {
        return told;
    };
    match event.kind {
        ResponsesEventType::ResponseCompleted | ResponsesEventType::ResponseIncomplete => {
            told.counting(event.response.unwrap_or_default().usage)
        }
        ResponsesEventType::ResponseFailed | ResponsesEventType::Error => {
            told.with_error(RESPONSES_ERROR_STATUS, data)
        }
        _ => told,
    }
}

/// How a stream went: the tokens it counted, or the failure that ended it.
pub(crate) type Outcome = Result<u64, AttemptError>;

fn lost_connection() -> AttemptError {
    AttemptError::new(FailureKind::ApiConnection, None)
}

/// A stream that broke under the client, as router-core reads it.
pub(crate) fn broke(error: &litellm_host_http::Error<InferenceError>) -> AttemptError {
    match error {
        litellm_host_http::Error::Call(route) => {
            routing::route_failure(route, lost_connection).unwrap_or_else(lost_connection)
        }
        litellm_host_http::Error::Protocol => lost_connection(),
    }
}

struct Watched<E> {
    blocks: BoxStream<'static, Result<SseBlock, E>>,
    /// What a stream whose outcome is not known yet said so far.
    told: Option<Told>,
}

/// `blocks` as they are, with `report` told how the stream went: once, and never for a stream
/// that is dropped before its outcome is known.
pub(crate) fn watched<E, R, F>(
    blocks: BoxStream<'static, Result<SseBlock, E>>,
    read: Read,
    broke: fn(&E) -> AttemptError,
    report: R,
) -> BoxStream<'static, Result<SseBlock, E>>
where
    E: Send + 'static,
    R: Fn(Outcome) -> F + Send + Sync + 'static,
    F: Future<Output = ()> + Send,
{
    let watched = Watched {
        blocks,
        told: Some(Told::default()),
    };
    stream::unfold((watched, report), move |(watched, report)| async move {
        let Watched { mut blocks, told } = watched;
        let block = blocks.next().await;
        let (told, outcome) = match (&block, told) {
            (_, None) => (None, None),
            (Some(Ok(block)), Some(told)) => {
                let told = match &block.event {
                    Some(event) => read(told, &event.data),
                    None => told,
                };
                match &told.error {
                    Some((status, sent)) => (
                        None,
                        Some(Err(AttemptError::from_provider(*status, sent, None))),
                    ),
                    None => (Some(told), None),
                }
            }
            (Some(Err(error)), Some(_)) => (None, Some(Err(broke(error)))),
            (None, Some(told)) => (None, Some(Ok(told.counts.total()))),
        };
        if let Some(outcome) = outcome {
            report(outcome).await;
        }
        Some((block?, (Watched { blocks, told }, report)))
    })
    .boxed()
}

/// A stream's failure after its attempt returned it, recorded against `deployment` as Python's
/// `report_stream_failure` records it: counted, and cooled down by the rules for the
/// deployment's own `cooldown_time`.
async fn record_failure<C: RouterCache>(
    router: &Router,
    cache: &C,
    deployment: &Deployment,
    failure: &AttemptError,
) {
    let cooldown_time = deployment.model_info.cooldown_time.or_else(|| {
        deployment
            .raw_params
            .get("cooldown_time")
            .and_then(Value::as_f64)
    });
    let report = FailureReport {
        deployment_cooldown: true,
        cooldown_time,
        ..FailureReport::default()
    };
    if let Err(error) = router
        .report_failure(cache, &deployment.id, failure, report)
        .await
    {
        tracing::warn!(deployment_id = %deployment.id, %error, "stream failure was not recorded");
    }
}

/// `chunks` of a stream `deployment` serves, as the blocks they carry, the stream's outcome
/// reported to the router.
pub(crate) fn reporting_outcome<H: routing::Host>(
    gateway: &Gateway<H>,
    deployment: &Arc<Deployment>,
    chunks: BoxStream<'static, Result<Bytes, litellm_host_http::Error<InferenceError>>>,
    read: Read,
) -> BoxStream<'static, Result<SseBlock, litellm_host_http::Error<InferenceError>>> {
    let router = gateway.router.clone();
    let routing = gateway.routing.clone();
    let deployment = deployment.clone();
    watched(
        sse::blocks(chunks).boxed(),
        read,
        broke,
        move |outcome: Outcome| {
            let (router, routing, deployment) =
                (router.clone(), routing.clone(), deployment.clone());
            async move {
                let total_tokens = match outcome {
                    Ok(total_tokens) => total_tokens,
                    Err(failure) => {
                        return record_failure(&router, &*routing, &deployment, &failure).await;
                    }
                };
                let recorded = router
                    .report_success(
                        &*routing,
                        &deployment.id,
                        total_tokens as f64,
                        0.0,
                        SuccessReport::default(),
                    )
                    .await;
                if let Err(error) = recorded {
                    tracing::warn!(deployment_id = %deployment.id, %error, "stream outcome was not recorded");
                }
            }
        },
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use futures_util::stream;
    use parking_lot::Mutex;
    use rstest::rstest;

    use super::*;

    const START: &str = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":7,\"output_tokens\":1}}}\n\n";
    const DELTA: &str = "event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{},\"usage\":{\"output_tokens\":5}}\n\n";
    const STOP: &str = "event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
    const OVERLOADED: &str = "event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\",\"message\":\"busy\"}}\n\n";
    const COMPLETED: &str = "event: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":4,\"total_tokens\":9}}}\n\n";
    const FAILED: &str = "event: response.failed\ndata: {\"type\":\"response.failed\",\"response\":{\"status\":\"failed\"}}\n\n";

    #[derive(Debug, PartialEq)]
    struct Dropped;

    fn dropped(_: &Dropped) -> AttemptError {
        AttemptError::new(FailureKind::ApiConnection, None)
    }

    /// What `report` heard, as the tokens of a success or the status and kind of a failure.
    type Heard = Result<u64, (Option<u16>, FailureKind)>;

    async fn heard(
        chunks: Vec<Result<&'static str, Dropped>>,
        read: Read,
        take: usize,
    ) -> (usize, Vec<Heard>) {
        let reports = Arc::new(Mutex::new(Vec::new()));
        let sink = reports.clone();
        let chunks =
            stream::iter(chunks).map(|chunk| chunk.map(|text| Bytes::from_static(text.as_bytes())));
        let blocks = sse::blocks(chunks).boxed();
        let delivered = watched(blocks, read, dropped, move |outcome: Outcome| {
            sink.lock()
                .push(outcome.map_err(|failure| (failure.status, failure.kind)));
            async {}
        })
        .take(take)
        .count()
        .await;
        let reports = reports.lock().clone();
        (delivered, reports)
    }

    #[rstest]
    #[case::messages_counts_input_and_final_output(
        vec![Ok(START), Ok(DELTA), Ok(STOP)],
        messages_event as Read,
        vec![Ok(12)]
    )]
    #[case::responses_counts_the_closing_total(
        vec![Ok(COMPLETED)],
        responses_event as Read,
        vec![Ok(9)]
    )]
    #[case::a_stream_without_usage_counts_nothing(vec![Ok(STOP)], messages_event as Read, vec![Ok(0)])]
    #[case::an_overloaded_messages_provider_fails_once_as_its_529(
        vec![Ok(START), Ok(OVERLOADED), Ok(STOP)],
        messages_event as Read,
        vec![Err((Some(529), FailureKind::InternalServer))]
    )]
    #[case::responses_failed_event_fails(
        vec![Ok(FAILED)],
        responses_event as Read,
        vec![Err((Some(500), FailureKind::InternalServer))]
    )]
    #[case::a_break_fails_once_and_the_rest_passes_through(
        vec![Ok(START), Err(Dropped), Ok(STOP)],
        messages_event as Read,
        vec![Err((None, FailureKind::ApiConnection))]
    )]
    #[tokio::test]
    async fn a_stream_read_to_its_end_reports_how_it_went_once(
        #[case] chunks: Vec<Result<&'static str, Dropped>>,
        #[case] read: Read,
        #[case] expected: Vec<Heard>,
    ) {
        let sent = chunks.len();

        assert_eq!(heard(chunks, read, usize::MAX).await, (sent, expected));
    }

    /// A provider's stream goes out as it was sent, so its reader counts events Anthropic itself
    /// would never send: a strict reading would lose the usage or the failure they carry.
    #[rstest]
    #[case::a_start_short_of_every_field_but_its_usage_counts_its_input(
        r#"{"type":"message_start","message":{"usage":{"input_tokens":7}}}"#,
        (7, None)
    )]
    #[case::an_error_without_a_message_fails_the_stream(
        r#"{"type":"error","error":{"type":"overloaded_error"}}"#,
        (0, Some(529))
    )]
    fn a_messages_event_a_host_sends_short_of_a_field_is_still_read(
        #[case] data: &str,
        #[case] expected: (u64, Option<u16>),
    ) {
        let told = messages_event(Told::default(), data);

        assert_eq!(
            (told.counts.total(), told.error.map(|(status, _)| status)),
            expected
        );
    }

    #[tokio::test]
    async fn an_event_split_across_chunks_is_read_whole() {
        let (head, tail) = DELTA.split_at(40);

        assert_eq!(
            heard(
                vec![Ok(START), Ok(head), Ok(tail)],
                messages_event,
                usize::MAX
            )
            .await,
            (2, vec![Ok(12)])
        );
    }

    #[tokio::test]
    async fn a_stream_the_client_stops_reading_reports_nothing() {
        assert_eq!(
            heard(vec![Ok(START), Ok(DELTA), Ok(STOP)], messages_event, 2).await,
            (2, vec![])
        );
    }
}
