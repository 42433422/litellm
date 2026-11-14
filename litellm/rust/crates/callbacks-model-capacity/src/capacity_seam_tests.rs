use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{Number, Value, json};

use litellm_callbacks_parallel_request_limiter::{Ctx, PolicyFacts};

use super::*;
use crate::{
    model::{Status, TokenRateLimitType},
    seam::{SeamFuture, ShareGauge},
    state::Stash,
};

struct FakeLimiter {
    redis: Result<Option<Value>, String>,
    local: Option<Value>,
    atomic: LimiterResponse,
    ttls: Mutex<Vec<(Counter, Value)>>,
    atomic_keys: Mutex<Vec<Vec<String>>>,
}

impl FakeLimiter {
    fn new(
        redis: Result<Option<Value>, String>,
        local: Option<Value>,
        atomic: LimiterResponse,
    ) -> Self {
        Self {
            redis,
            local,
            atomic,
            ttls: Mutex::new(Vec::new()),
            atomic_keys: Mutex::new(Vec::new()),
        }
    }
}

impl V3Limiter for FakeLimiter {
    fn read_counter<'a>(
        &'a self,
        counter: &'a Counter,
        backfill_ttl: &'a Value,
    ) -> SeamFuture<'a, Option<Value>> {
        self.ttls
            .lock()
            .push((counter.clone(), backfill_ttl.clone()));
        let reply = self.redis.clone().map_err(CapacityError::Store);
        Box::pin(async move { reply })
    }

    fn read_local_counter(&self, _counter: &Counter) -> Result<Option<Value>, CapacityError> {
        Ok(self.local.clone())
    }

    fn check_and_increment<'a>(
        &'a self,
        descriptors: &'a [Descriptor],
        _increments: &'a [Increment],
        _ctx: Ctx<'a>,
    ) -> SeamFuture<'a, LimiterResponse> {
        self.atomic_keys.lock().push(
            descriptors
                .iter()
                .map(|descriptor| descriptor.key.clone())
                .collect(),
        );
        let reply = self.atomic.clone();
        Box::pin(async move { Ok(reply) })
    }

    fn should_rate_limit<'a>(
        &'a self,
        _descriptors: &'a [Descriptor],
        _read_only: bool,
        _skip_parallel_check: bool,
        _ctx: Ctx<'a>,
    ) -> SeamFuture<'a, LimiterResponse> {
        Box::pin(async move { Ok(answer("OK", Vec::new())) })
    }

    fn apply<'a>(&'a self, _operations: &'a [ReservationOp]) -> SeamFuture<'a, ()> {
        Box::pin(async move { Ok(()) })
    }
}

#[derive(Default)]
struct FakeHost {
    group: Option<GroupInfo>,
    logged: Mutex<Vec<(String, String)>>,
}

fn done<'a, T: Send + 'a>(value: T) -> SeamFuture<'a, T> {
    Box::pin(async move { Ok(value) })
}

impl CapacityHost for FakeHost {
    fn settings(&self) -> SeamFuture<'_, Settings> {
        done(settings(json!(null)))
    }

    fn group<'a>(&'a self, _model: &'a str) -> SeamFuture<'a, Option<GroupInfo>> {
        done(self.group.clone())
    }

    fn touch<'a>(
        &'a self,
        _model: &'a str,
        _member: &'a str,
        _priority_class: &'a Value,
        _window: i64,
    ) -> SeamFuture<'a, Vec<(String, Value)>> {
        done(Vec::new())
    }

    fn metrics_enabled(&self) -> SeamFuture<'_, bool> {
        done(false)
    }

    fn throttled<'a>(&'a self, _event: &'a Throttle) -> SeamFuture<'a, ()> {
        done(())
    }

    fn set_share<'a>(&'a self, _gauge: &'a ShareGauge) -> SeamFuture<'a, ()> {
        done(())
    }

    fn unlicensed_priority(&self) -> SeamFuture<'_, ()> {
        done(())
    }

    fn log<'a>(&'a self, level: &'static str, message: &'a str) -> SeamFuture<'a, ()> {
        self.logged
            .lock()
            .push((level.to_owned(), message.to_owned()));
        done(())
    }

    fn borrowing_limits<'a>(&'a self, _model: &'a str) -> SeamFuture<'a, Option<BorrowingLimits>> {
        done(None)
    }

    fn model_group<'a>(&'a self, model: &'a str) -> SeamFuture<'a, String> {
        done(model.to_owned())
    }

    fn estimate_tokens<'a>(
        &'a self,
        _tpm_share: i64,
        _model: &'a str,
        _call_type: Option<&'a str>,
    ) -> SeamFuture<'a, Estimate> {
        done(Estimate::NoBody)
    }

    fn cap_output<'a>(
        &'a self,
        _tpm_share: i64,
        _call_type: Option<&'a str>,
    ) -> SeamFuture<'a, ()> {
        done(())
    }

    fn publish<'a>(&'a self, _published: &'a Published) -> SeamFuture<'a, ()> {
        done(())
    }

    fn stash_call_type(&self) -> SeamFuture<'_, Option<String>> {
        done(None)
    }

    fn post_call_success<'a>(&'a self, _headers: &'a [(String, Value)]) -> SeamFuture<'a, ()> {
        done(())
    }
}

fn settings(ttl: Value) -> Settings {
    Settings {
        priority_reservation: Some(vec![
            (json!("high"), json!(0.9)),
            (json!("low"), json!(0.1)),
        ]),
        default_priority: 0.5,
        saturation_threshold: 0.5,
        class_split: false,
        saturation_check_cache_ttl: ttl,
        window_size: 60,
        rate_limit_type: TokenRateLimitType::default(),
        licensed: true,
        policy: PolicyFacts::default(),
    }
}

fn group(tpm: Option<i64>, rpm: Option<i64>) -> GroupInfo {
    GroupInfo {
        tpm,
        rpm,
        input_cost_per_token: None,
        output_cost_per_token: None,
    }
}

fn answer(code: &str, statuses: Vec<Status>) -> LimiterResponse {
    LimiterResponse {
        overall_code: code.into(),
        statuses,
        reservation_windows: Vec::new(),
    }
}

fn engine(limiter: &Arc<FakeLimiter>, host: &Arc<FakeHost>) -> Engine {
    Engine {
        limiter: Arc::clone(limiter) as Arc<dyn V3Limiter>,
        host: Arc::clone(host) as Arc<dyn CapacityHost>,
    }
}

async fn pre_call(engine: &Engine, settings: &Settings, model: &str, group: &GroupInfo) -> PreCall {
    let key = KeyFacts {
        priority: Some(Priority {
            value: json!("high"),
            hash_error: None,
        }),
        ..KeyFacts::default()
    };
    let stash: SharedStash = Arc::new(Mutex::new(Stash::default()));
    let request = Request {
        key: &key,
        call_type: None,
        settings,
        stash: &stash,
        fallback: SaturationFallback {
            enabled: false,
            invalid: None,
            armed: false,
        },
        redis: None,
    };
    engine.pre_call(&request, model, group).await
}

#[tokio::test]
async fn every_saturation_read_backfills_for_the_configured_cache_ttl_like_python() {
    let limiter = Arc::new(FakeLimiter::new(Ok(None), None, answer("OK", Vec::new())));
    let host = Arc::new(FakeHost::default());
    let engine = engine(&limiter, &host);
    let model_group = group(Some(1000), Some(100));
    for ttl in [5, 30] {
        engine
            .saturation(&settings(json!(ttl)), "test-saturation-ttl", &model_group)
            .await;
    }
    let requests =
        scope(MODEL_SATURATION_CHECK, "test-saturation-ttl").with_kind(CounterKind::Requests);
    let read: Vec<Value> = limiter
        .ttls
        .lock()
        .iter()
        .filter(|(counter, _)| *counter == requests)
        .map(|(_, ttl)| ttl.clone())
        .collect();
    assert_eq!(read, vec![json!(5), json!(30)]);
    assert!(
        limiter
            .ttls
            .lock()
            .iter()
            .all(|(_, ttl)| *ttl == json!(5) || *ttl == json!(30))
    );
}

#[tokio::test]
async fn a_failed_saturation_read_falls_back_to_this_workers_copy_like_python() {
    let limiter = Arc::new(FakeLimiter::new(
        Err("redis down".into()),
        Some(json!(3)),
        answer("OK", Vec::new()),
    ));
    let host = Arc::new(FakeHost::default());
    let saturation = engine(&limiter, &host)
        .saturation(&settings(json!(60)), "down", &group(None, Some(4)))
        .await;
    assert!((saturation - 0.75).abs() < f64::EPSILON);
    assert_eq!(
        *host.logged.lock(),
        vec![(
            "warning".to_owned(),
            "Saturation read failed in Redis, using this worker's copy: redis down".to_owned()
        )]
    );
}

#[tokio::test]
async fn enforced_descriptors_go_through_the_atomic_check_like_python() {
    let limiter = Arc::new(FakeLimiter::new(Ok(None), None, answer("OK", Vec::new())));
    let model_group = group(Some(1000), None);
    let host = Arc::new(FakeHost {
        group: Some(model_group.clone()),
        ..FakeHost::default()
    });
    let outcome = pre_call(
        &engine(&limiter, &host),
        &settings(json!(60)),
        "atomic-dyn-model",
        &model_group,
    )
    .await;
    assert!(matches!(outcome, PreCall::Allow));
    let atomic_keys = limiter.atomic_keys.lock();
    assert!(
        atomic_keys
            .first()
            .is_some_and(|keys| keys.iter().any(|key| key == MODEL_SATURATION_CHECK)),
        "{atomic_keys:?}"
    );
}

#[tokio::test]
async fn an_over_limit_answer_for_an_unknown_descriptor_is_refused_like_python() {
    let unknown = Status {
        code: "OVER_LIMIT".into(),
        current_limit: Number::from(100),
        limit_remaining: Some(Number::from(0)),
        rate_limit_type: "tokens".into(),
        descriptor_key: "future_unrecognized_descriptor".into(),
        descriptor_value: None,
    };
    let limiter = Arc::new(FakeLimiter::new(
        Ok(None),
        None,
        answer("OVER_LIMIT", vec![unknown]),
    ));
    let model_group = group(Some(1000), None);
    let host = Arc::new(FakeHost {
        group: Some(model_group.clone()),
        ..FakeHost::default()
    });
    match pre_call(
        &engine(&limiter, &host),
        &settings(json!(60)),
        "fail-closed-model",
        &model_group,
    )
    .await
    {
        PreCall::Refuse(refusal) => assert_eq!(refusal.model, "fail-closed-model"),
        _ => panic!("an unknown over-limit descriptor must be refused"),
    }
}
