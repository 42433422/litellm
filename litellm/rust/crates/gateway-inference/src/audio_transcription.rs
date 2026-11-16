use litellm_gateway_auth::AuthenticatedRequest;
use std::{path::Path, sync::Arc};

use axum::{Json, extract::State, http::HeaderMap, response::IntoResponse};
use base64::{Engine, engine::general_purpose::STANDARD};
use litellm_inference_transcription::{AudioTranscriptionRoute, types::AudioTranscriptionRequest};
use litellm_router_core::{Attempt, Op, Served};
use serde_json::{Map, Value, json};

use crate::{
    Error, Gateway,
    request::{self, InferenceBody},
    router_host::{self, RouteAttempt},
};

pub(crate) async fn create(
    State(gateway): State<Arc<Gateway>>,
    identity: AuthenticatedRequest,
    body: InferenceBody,
) -> Result<impl IntoResponse, Error> {
    handle(&gateway, &identity, body)
        .await
        .map(|(headers, body)| (headers, Json(body)))
}

async fn handle(
    gateway: &Gateway,
    identity: &AuthenticatedRequest,
    InferenceBody {
        fields: body,
        upload,
    }: InferenceBody,
) -> Result<(HeaderMap, Value), Error> {
    let model = request::model(&body)?.to_owned();
    let audio = match upload {
        Some(upload) => {
            let format = upload
                .file_name
                .as_deref()
                .and_then(|name| Path::new(name).extension())
                .and_then(|extension| extension.to_str())
                .map(str::to_ascii_lowercase);
            json!({"data": STANDARD.encode(upload.bytes), "format": format})
        }
        None => body.get("audio").cloned().unwrap_or_default(),
    };
    let attempt = TranscriptionAttempt {
        route: &gateway.audio_transcription,
        body,
        audio,
    };
    let routed = router_host::call(
        gateway,
        identity,
        Op::TRANSCRIPTION,
        "/audio/transcriptions",
        &model,
        &attempt,
    )
    .await?;
    Ok((router_host::headers(&routed), routed.output.response))
}

struct TranscriptionAttempt<'a> {
    route: &'a AudioTranscriptionRoute,
    body: Map<String, Value>,
    audio: Value,
}

impl RouteAttempt for TranscriptionAttempt<'_> {
    type Outcome = Value;

    async fn run(&self, attempt: &Attempt) -> Result<Served<Value>, Error> {
        let deployment = &attempt.deployment;
        let api_key = router_host::api_key(deployment);
        let body = self
            .route
            .execute(AudioTranscriptionRequest {
                model: &deployment.params.model,
                audio: self.audio.clone(),
                api_key: api_key.as_deref(),
                api_base: deployment.params.api_base.as_deref(),
                custom_llm_provider: deployment.params.custom_llm_provider.as_deref(),
                extra_headers: None,
                optional_params: router_host::attempt_body(&self.body, attempt)
                    .into_iter()
                    .filter(|(name, _)| !matches!(name.as_str(), "model" | "audio"))
                    .collect(),
                timeout: router_host::timeout(attempt),
            })
            .await?;
        let total_tokens = router_host::total_tokens(body.get("usage"));
        Ok(Served::new(body, total_tokens))
    }
}
