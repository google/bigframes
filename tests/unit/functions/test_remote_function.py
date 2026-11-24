# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random
import unittest.mock as mock

import google.api_core.exceptions
import pytest
from google.cloud import bigquery, functions_v2

from bigframes.functions import _function_client, _function_session, _utils, udf_def
from bigframes.testing import mocks


def test_missing_input_types():
    session = mocks.create_bigquery_session()
    remote_function_decorator = session._function_session.remote_function(
        cloud_function_service_account="default"
    )

    def function_without_parameter_annotations(myparam) -> str:
        return str(myparam)

    assert function_without_parameter_annotations(42) == "42"

    with pytest.raises(
        ValueError,
        match="'input_types' was not set .* 'myparam' is missing a type annotation",
    ):
        remote_function_decorator(function_without_parameter_annotations)


def test_missing_output_type():
    session = mocks.create_bigquery_session()
    remote_function_decorator = session._function_session.remote_function(
        cloud_function_service_account="default"
    )

    def function_without_return_annotation(myparam: int):
        return str(myparam)

    assert function_without_return_annotation(42) == "42"

    with pytest.raises(
        ValueError,
        match="'output_type' was not set .* missing a return type annotation",
    ):
        remote_function_decorator(function_without_return_annotation)


def test_deploy_udf():
    session = mocks.create_bigquery_session()

    def my_remote_func(x: int) -> int:
        return x * 2

    deployed = session.deploy_udf(my_remote_func)

    assert deployed.udf_def is not None


def test_deploy_udf_with_name():
    session = mocks.create_bigquery_session()

    def my_remote_func(x: int) -> int:
        return x * 2

    deployed = session.deploy_udf(my_remote_func, name="my_custom_name")

    # Test that the function would have been deployed somewhere.
    assert "my_custom_name" in deployed.bigframes_bigquery_function


def _sample_cloud_run_config(
    func=None,
    *,
    packages=(),
    ingress_settings="internal-only",
    cloud_run_service_account="alice-sa@test-project.iam.gserviceaccount.com",
    kms_key_name=None,
    docker_repository=None,
    vpc_connector=None,
    vpc_connector_egress_settings="private-ranges-only",
    cloud_build_service_account=None,
) -> udf_def.CloudRunFunctionConfig:
    if func is None:

        def func(x: int) -> int:
            return x + 1

    sig = _utils.get_func_signature(func).to_remote_function_compatible()
    return udf_def.CloudRunFunctionConfig(
        code=udf_def.CodeDef.from_func(func, packages),
        signature=sig,
        timeout_seconds=600,
        max_instance_count=None,
        vpc_connector=vpc_connector,
        vpc_connector_egress_settings=vpc_connector_egress_settings,
        memory_mib=1024,
        cpus=None,
        ingress_settings=ingress_settings,
        workers=1,
        threads=4,
        concurrency=1,
        kms_key_name=kms_key_name,
        docker_repository=docker_repository,
        cloud_build_service_account=cloud_build_service_account,
        cloud_run_service_account=cloud_run_service_account,
    )


def _matching_gcf_resource(
    config: udf_def.CloudRunFunctionConfig,
    *,
    uri: str = "https://example-cf.a.run.app",
    labels: dict | None = None,
    runtime: str | None = None,
    ingress_settings=None,
    service_account_email: str | None = None,
    kms_key_name: str | None = None,
    vpc_connector: str | None = None,
    vpc_connector_egress_settings=None,
    docker_repository: str | None = None,
    build_service_account: str | None = None,
) -> functions_v2.Function:
    func = functions_v2.Function()
    func.build_config = functions_v2.BuildConfig()
    func.build_config.runtime = runtime or _utils.get_python_version(is_compat=True)
    if docker_repository is not None:
        func.build_config.docker_repository = docker_repository
    elif config.docker_repository is not None:
        func.build_config.docker_repository = config.docker_repository
    if build_service_account is not None:
        func.build_config.service_account = build_service_account
    elif config.cloud_build_service_account is not None:
        func.build_config.service_account = config.cloud_build_service_account

    func.service_config = functions_v2.ServiceConfig()
    func.service_config.uri = uri
    func.service_config.ingress_settings = (
        ingress_settings
        if ingress_settings is not None
        else _function_client._INGRESS_SETTINGS_MAP[config.ingress_settings]
    )
    if service_account_email is not None:
        func.service_config.service_account_email = service_account_email
    elif config.cloud_run_service_account is not None:
        func.service_config.service_account_email = config.cloud_run_service_account

    if vpc_connector is not None:
        func.service_config.vpc_connector = vpc_connector
    elif config.vpc_connector is not None:
        func.service_config.vpc_connector = config.vpc_connector
    if vpc_connector_egress_settings is not None:
        func.service_config.vpc_connector_egress_settings = (
            vpc_connector_egress_settings
        )
    elif config.vpc_connector is not None:
        func.service_config.vpc_connector_egress_settings = (
            _function_client._VPC_EGRESS_SETTINGS_MAP[
                config.vpc_connector_egress_settings
            ]
        )

    if kms_key_name is not None:
        func.kms_key_name = kms_key_name
    elif config.kms_key_name is not None:
        func.kms_key_name = config.kms_key_name

    if labels is not None:
        func.labels = labels

    return func


def _make_function_client(mock_gcf_client=None) -> _function_client.FunctionClient:
    if mock_gcf_client is None:
        mock_gcf_client = mock.create_autospec(
            functions_v2.FunctionServiceClient, instance=True
        )
    mock_gcf_client.common_location_path.side_effect = (
        lambda project, location: f"projects/{project}/locations/{location}"
    )
    mock_gcf_client.function_path.side_effect = lambda project, location, name: (
        f"projects/{project}/locations/{location}/functions/{name}"
    )
    mock_bq_client = mock.create_autospec(bigquery.Client, instance=True)
    return _function_client.FunctionClient(
        gcp_project_id="test-project",
        bq_location="us",
        bq_client=mock_bq_client,
        bq_connection_manager=mock.MagicMock(),
        cloud_functions_client=mock_gcf_client,
        publisher=mock.MagicMock(),
    )


def _make_upload_url_response() -> functions_v2.GenerateUploadUrlResponse:
    upload_resp = functions_v2.GenerateUploadUrlResponse()
    upload_resp.upload_url = "https://storage.googleapis.com/upload-target"
    upload_resp.storage_source.bucket = "bucket"
    upload_resp.storage_source.object_ = "obj.zip"
    return upload_resp


def _make_mock_function_client(session) -> mock.MagicMock:
    mock_client = mock.create_autospec(_function_client.FunctionClient, instance=True)
    mock_client._bq_client = session.bqclient
    mock_client.get_cloud_function_fully_qualified_name.side_effect = (
        lambda name: f"projects/test-project/locations/us-central1/functions/{name}"
    )
    return mock_client


def test_stable_hash_returns_deterministic_128bit_digest():
    config = _sample_cloud_run_config()

    first_digest = config.stable_hash()
    second_digest = config.stable_hash()

    assert len(first_digest) == 16
    assert len(first_digest.hex()) == 32
    assert len(config.code.stable_hash()) == 16
    assert len(config.signature.stable_hash()) == 16
    assert first_digest == second_digest


def test_get_cloud_function_name_fits_gcp_length_limit():
    config = _sample_cloud_run_config()
    expected_hex = config.stable_hash().hex()

    cf_name = _function_session.get_cloud_function_name(
        config, session_id="sessionabcdef", unique_suffix="a1b2"
    )

    assert len(cf_name) <= 63
    assert cf_name == f"bigframes-sessionabcdef-{expected_hex}-a1b2"


def test_code_def_stable_hash_delimits_package_requirements():
    def fn(x: int) -> int:
        return x + 1

    code_a = udf_def.CodeDef.from_func(fn, ["ab", "c"])
    code_b = udf_def.CodeDef.from_func(fn, ["a", "bc"])

    hash_a = code_a.stable_hash()
    hash_b = code_b.stable_hash()

    assert hash_a != hash_b


def test_runtime_requirements_stable_hash_distinguishes_none_from_string():
    req_none = udf_def.RuntimeRequirements(container_memory=None)
    req_str = udf_def.RuntimeRequirements(container_memory="None")

    hash_none = req_none.stable_hash()
    hash_str = req_str.stable_hash()

    assert hash_none != hash_str


def test_cloud_run_config_stable_hash_distinguishes_ingress_settings():
    def fn(x: int) -> int:
        return x + 1

    cfg_internal = _sample_cloud_run_config(fn, ingress_settings="internal-only")
    cfg_all = _sample_cloud_run_config(fn, ingress_settings="all")

    hash_internal = cfg_internal.stable_hash()
    hash_all = cfg_all.stable_hash()

    assert hash_internal != hash_all


def test_cloud_run_config_stable_hash_distinguishes_service_account():
    def fn(x: int) -> int:
        return x + 1

    cfg_sa1 = _sample_cloud_run_config(fn, cloud_run_service_account="sa1@proj.iam")
    cfg_sa2 = _sample_cloud_run_config(fn, cloud_run_service_account="sa2@proj.iam")

    hash_sa1 = cfg_sa1.stable_hash()
    hash_sa2 = cfg_sa2.stable_hash()

    assert hash_sa1 != hash_sa2


def test_get_cloud_function_endpoint_returns_uri_when_config_matches():
    mock_gcf_client = mock.create_autospec(
        functions_v2.FunctionServiceClient, instance=True
    )
    client = _make_function_client(mock_gcf_client)
    config = _sample_cloud_run_config(
        ingress_settings="internal-only",
        cloud_run_service_account="alice-sa@test-project.iam.gserviceaccount.com",
        kms_key_name="projects/test-project/locations/us-central1/keyRings/r/cryptoKeys/k",
        vpc_connector="projects/test-project/locations/us-central1/connectors/c",
    )
    mock_gcf_client.get_function.return_value = _matching_gcf_resource(config)

    endpoint = client.get_cloud_function_endpoint(
        "bigframes-test", expected_config=config
    )

    assert endpoint == "https://example-cf.a.run.app"


@pytest.mark.parametrize(
    "mismatched_overrides",
    [
        pytest.param(
            {"ingress_settings": functions_v2.ServiceConfig.IngressSettings.ALLOW_ALL},
            id="ingress_settings",
        ),
        pytest.param(
            {"service_account_email": "attacker@test-project.iam.gserviceaccount.com"},
            id="service_account_email",
        ),
        pytest.param(
            {"kms_key_name": ""},
            id="kms_key_name",
        ),
        pytest.param(
            {"vpc_connector": ""},
            id="vpc_connector",
        ),
        pytest.param(
            {"runtime": "nodejs20"},
            id="runtime",
        ),
    ],
)
def test_get_cloud_function_endpoint_rejects_mismatched_config(mismatched_overrides):
    mock_gcf_client = mock.create_autospec(
        functions_v2.FunctionServiceClient, instance=True
    )
    client = _make_function_client(mock_gcf_client)
    config = _sample_cloud_run_config(
        ingress_settings="internal-only",
        cloud_run_service_account="alice-sa@test-project.iam.gserviceaccount.com",
        kms_key_name="projects/test-project/locations/us-central1/keyRings/r/cryptoKeys/k",
        vpc_connector="projects/test-project/locations/us-central1/connectors/c",
    )
    mock_gcf_client.get_function.return_value = _matching_gcf_resource(
        config, **mismatched_overrides
    )

    with pytest.raises(ValueError, match="does not match the requested configuration"):
        client.get_cloud_function_endpoint("bigframes-test", expected_config=config)


def test_get_cloud_function_endpoint_waits_for_pending_uri():
    mock_gcf_client = mock.create_autospec(
        functions_v2.FunctionServiceClient, instance=True
    )
    client = _make_function_client(mock_gcf_client)
    config = _sample_cloud_run_config()
    pending_fn = _matching_gcf_resource(config, uri="")
    ready_fn = _matching_gcf_resource(config, uri="https://ready-cf.a.run.app")
    mock_gcf_client.get_function.side_effect = [pending_fn, ready_fn]

    endpoint = client.get_cloud_function_endpoint(
        "bigframes-test", expected_config=config
    )

    assert endpoint == "https://ready-cf.a.run.app"
    assert mock_gcf_client.get_function.call_count == 2


def test_create_cloud_function_allows_own_retry_via_deploy_nonce():
    mock_gcf_client = mock.create_autospec(
        functions_v2.FunctionServiceClient, instance=True
    )
    client = _make_function_client(mock_gcf_client)
    config = _sample_cloud_run_config()
    mock_gcf_client.generate_upload_url.return_value = _make_upload_url_response()
    captured_nonce = {}

    def fake_create_function(*, request):
        captured_nonce["value"] = request.function.labels[
            _function_client._DEPLOY_NONCE_LABEL_KEY
        ]
        raise google.api_core.exceptions.AlreadyExists("already exists from retry")

    mock_gcf_client.create_function.side_effect = fake_create_function
    mock_gcf_client.get_function.side_effect = lambda *, name: _matching_gcf_resource(
        config,
        uri="https://retried-own-cf.a.run.app",
        labels={_function_client._DEPLOY_NONCE_LABEL_KEY: captured_nonce["value"]},
    )
    mock_put_resp = mock.MagicMock(status_code=200)

    with mock.patch(
        "bigframes.functions._function_client.requests.put", return_value=mock_put_resp
    ):
        endpoint = client.create_cloud_function(
            "bigframes-test", config, maybe_reuse=False
        )

    assert endpoint == "https://retried-own-cf.a.run.app"
    assert len(captured_nonce["value"]) == 32


def test_create_cloud_function_rejects_foreign_already_exists_when_reuse_false():
    mock_gcf_client = mock.create_autospec(
        functions_v2.FunctionServiceClient, instance=True
    )
    client = _make_function_client(mock_gcf_client)
    config = _sample_cloud_run_config()
    mock_gcf_client.generate_upload_url.return_value = _make_upload_url_response()
    mock_gcf_client.create_function.side_effect = (
        google.api_core.exceptions.AlreadyExists("pre-created by attacker")
    )
    mock_gcf_client.get_function.return_value = _matching_gcf_resource(
        config,
        uri="https://attacker-cf.a.run.app",
        labels={_function_client._DEPLOY_NONCE_LABEL_KEY: "0" * 32},
    )
    mock_put_resp = mock.MagicMock(status_code=200)

    with mock.patch(
        "bigframes.functions._function_client.requests.put", return_value=mock_put_resp
    ):
        with pytest.raises(google.api_core.exceptions.AlreadyExists):
            client.create_cloud_function("bigframes-test", config, maybe_reuse=False)


def test_create_cloud_function_adopts_matching_existing_on_reuse_true_conflict():
    mock_gcf_client = mock.create_autospec(
        functions_v2.FunctionServiceClient, instance=True
    )
    client = _make_function_client(mock_gcf_client)
    config = _sample_cloud_run_config(ingress_settings="internal-only")
    mock_gcf_client.generate_upload_url.return_value = _make_upload_url_response()
    mock_gcf_client.create_function.side_effect = (
        google.api_core.exceptions.AlreadyExists("concurrent worker created")
    )
    mock_gcf_client.get_function.return_value = _matching_gcf_resource(
        config,
        uri="https://concurrent-cf.a.run.app",
        labels={_function_client._DEPLOY_NONCE_LABEL_KEY: "other-worker-nonce"},
    )
    mock_put_resp = mock.MagicMock(status_code=200)

    with mock.patch(
        "bigframes.functions._function_client.requests.put", return_value=mock_put_resp
    ):
        endpoint = client.create_cloud_function(
            "bigframes-test", config, maybe_reuse=True
        )

    assert endpoint == "https://concurrent-cf.a.run.app"


def test_create_cloud_function_rejects_mismatched_existing_on_reuse_true_conflict():
    mock_gcf_client = mock.create_autospec(
        functions_v2.FunctionServiceClient, instance=True
    )
    client = _make_function_client(mock_gcf_client)
    config = _sample_cloud_run_config(ingress_settings="internal-only")
    mock_gcf_client.generate_upload_url.return_value = _make_upload_url_response()
    mock_gcf_client.create_function.side_effect = (
        google.api_core.exceptions.AlreadyExists("pre-created with weaker ingress")
    )
    mock_gcf_client.get_function.return_value = _matching_gcf_resource(
        config,
        uri="https://hijacked-cf.a.run.app",
        ingress_settings=functions_v2.ServiceConfig.IngressSettings.ALLOW_ALL,
        labels={},
    )
    mock_put_resp = mock.MagicMock(status_code=200)

    with mock.patch(
        "bigframes.functions._function_client.requests.put", return_value=mock_put_resp
    ):
        with pytest.raises(google.api_core.exceptions.AlreadyExists):
            client.create_cloud_function("bigframes-test", config, maybe_reuse=True)


def test_remote_function_reuse_false_skips_endpoint_lookup():
    session = mocks.create_bigquery_session()
    func_session = session._function_session
    mock_client = _make_mock_function_client(session)
    mock_client.create_cloud_function.side_effect = (
        lambda cf_name, spec, *, maybe_reuse=False: f"https://{cf_name}.a.run.app"
    )
    func_session._function_client = mock_client

    def my_fn(x: int) -> int:
        return x * 2

    func_session.remote_function(
        reuse=False,
        cloud_function_service_account="alice-sa@test-project.iam.gserviceaccount.com",
    )(my_fn)

    mock_client.get_cloud_function_endpoint.assert_not_called()
    mock_client.create_cloud_function.assert_called_once()
    _, kwargs = mock_client.create_cloud_function.call_args
    assert kwargs["maybe_reuse"] is False


def test_remote_function_reuse_false_retries_with_new_suffix_on_already_exists():
    session = mocks.create_bigquery_session()
    func_session = session._function_session
    mock_client = _make_mock_function_client(session)
    created_names = []

    def fake_create(cf_name, spec, *, maybe_reuse=False):
        created_names.append(cf_name)
        if len(created_names) == 1:
            raise google.api_core.exceptions.AlreadyExists("colliding foreign fn")
        return f"https://{cf_name}.a.run.app"

    mock_client.create_cloud_function.side_effect = fake_create
    func_session._function_client = mock_client

    def my_fn(x: int) -> int:
        return x * 2

    wrapped = func_session.remote_function(
        reuse=False,
        cloud_function_service_account="alice-sa@test-project.iam.gserviceaccount.com",
    )(my_fn)

    assert len(created_names) == 2
    assert created_names[0] != created_names[1]
    assert created_names[0] not in func_session._temp_cloud_functions
    assert created_names[1] in func_session._temp_cloud_functions
    assert wrapped.bigframes_cloud_function.endswith(f"/{created_names[1]}")


def test_remote_function_reuse_false_raises_after_exhausting_conflict_retries():
    session = mocks.create_bigquery_session()
    func_session = session._function_session
    mock_client = _make_mock_function_client(session)
    mock_client.create_cloud_function.side_effect = (
        google.api_core.exceptions.AlreadyExists("persistent conflict")
    )
    func_session._function_client = mock_client

    def my_fn(x: int) -> int:
        return x * 2

    with pytest.raises(google.api_core.exceptions.AlreadyExists):
        func_session.remote_function(
            reuse=False,
            cloud_function_service_account="alice-sa@test-project.iam.gserviceaccount.com",
        )(my_fn)

    assert (
        mock_client.create_cloud_function.call_count
        == _function_session._MAX_CLOUD_FUNCTION_CREATE_ATTEMPTS
    )
    assert len(func_session._temp_cloud_functions) == 0


def test_remote_function_reuse_false_not_deterministic_under_random_seed():
    session = mocks.create_bigquery_session()
    func_session = session._function_session
    mock_client = _make_mock_function_client(session)
    mock_client.create_cloud_function.side_effect = (
        lambda cf_name, spec, *, maybe_reuse=False: f"https://{cf_name}.a.run.app"
    )
    func_session._function_client = mock_client

    def my_fn(x: int) -> int:
        return x + 5

    generated = set()
    for _ in range(5):
        random.seed(12345)
        wrapped = func_session.remote_function(
            reuse=False,
            name="named_fn",
            cloud_function_service_account="alice-sa@test-project.iam.gserviceaccount.com",
        )(my_fn)
        generated.add(wrapped.bigframes_cloud_function)

    assert len(generated) > 1


def test_remote_function_reuse_true_passes_expected_config():
    session = mocks.create_bigquery_session()
    func_session = session._function_session
    mock_client = _make_mock_function_client(session)
    mock_client.get_cloud_function_endpoint.return_value = "https://existing.a.run.app"
    func_session._function_client = mock_client

    def my_fn(x: int) -> int:
        return x + 1

    func_session.remote_function(
        reuse=True,
        name="reused_fn",
        cloud_function_service_account="alice-sa@test-project.iam.gserviceaccount.com",
    )(my_fn)

    mock_client.get_cloud_function_endpoint.assert_called_once()
    _, kwargs = mock_client.get_cloud_function_endpoint.call_args
    assert isinstance(kwargs["expected_config"], udf_def.CloudRunFunctionConfig)
    mock_client.create_cloud_function.assert_not_called()
