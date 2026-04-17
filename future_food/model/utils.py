def generate_API_url(
        datablock,
        base_url="https://sarahjp-hack.streamlit.app/?",
        keys=None
        ):
    """Generates a URL for the current model run"""

    url = base_url

    if keys is None:
        keys = []
    for key in keys:
        url += f"{key}={datablock['run_params'][key]}&"

    datablock["URL"] = url

    return datablock