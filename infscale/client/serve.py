"""implementation of serve restapi call."""

import asyncio

import yaml
from infscale.actor.worker import Worker
from infscale.openapi import ApiClient, Configuration, DefaultApi, ServeSpec


def serve(host: str, port: int, specfile: str):
    """Call serve restapi."""
    endpoint = f"http://{host}:{port}"

    with open(specfile, "r") as f:
        spec_dict = yaml.safe_load(f)

    config = Configuration(endpoint)
    config.client_side_validation = False
    with ApiClient(config) as api_client:
        # Create an instance of the API class
        api_instance = DefaultApi(api_client)
        var_self = None
        spec = ServeSpec(**spec_dict)

        try:
            api_response = api_instance.serve_models_post(var_self, spec)
            print(f"{api_response.message}")
            # pprint(api_response)
        except Exception as e:
            print(f"Exception during serve api call: {e}")

    loop = asyncio.get_event_loop()
    loop.run_until_complete(Worker(0, None, spec_dict).run())
