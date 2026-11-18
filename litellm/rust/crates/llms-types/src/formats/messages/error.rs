/// Every error type on https://platform.claude.com/docs/en/api/errors as of 2026-11-18, the
/// `type` of the `error` in an error response and in a stream's `error` event. A type a host
/// adds rides in `Other` untouched.
#[derive(
    Clone,
    Debug,
    PartialEq,
    Eq,
    strum::EnumIter,
    strum::EnumString,
    strum::Display,
    serde_with::DeserializeFromStr,
    serde_with::SerializeDisplay,
)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(from = "String", into = "String"))]
#[strum(serialize_all = "snake_case")]
pub enum MessagesErrorType {
    InvalidRequestError,
    AuthenticationError,
    BillingError,
    PermissionError,
    NotFoundError,
    ConflictError,
    RequestTooLarge,
    RateLimitError,
    ApiError,
    TimeoutError,
    OverloadedError,
    #[strum(default, transparent)]
    Other(String),
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use serde_json::{Value, json};

    use super::*;

    #[rstest]
    #[case::suffixed(json!("overloaded_error"), MessagesErrorType::OverloadedError)]
    #[case::unsuffixed(json!("request_too_large"), MessagesErrorType::RequestTooLarge)]
    #[case::unknown(json!("future_error"), MessagesErrorType::Other("future_error".into()))]
    #[case::case_sensitive(json!("Api_Error"), MessagesErrorType::Other("Api_Error".into()))]
    fn an_error_type_round_trips_as_its_wire_spelling(
        #[case] wire: Value,
        #[case] expected: MessagesErrorType,
    ) {
        let read: MessagesErrorType = serde_json::from_value(wire.clone()).unwrap();

        assert_eq!(
            (&read, serde_json::to_value(&read).unwrap()),
            (&expected, wire)
        );
    }

    #[rstest]
    fn a_type_that_is_not_text_is_rejected(
        #[values(json!(529), json!(null), json!({}))] wire: Value,
    ) {
        assert!(serde_json::from_value::<MessagesErrorType>(wire).is_err());
    }
}
