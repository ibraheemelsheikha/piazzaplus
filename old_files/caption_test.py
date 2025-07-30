import base64
import httpx
from langchain_openai import ChatOpenAI
import requests

def to_cdn_url(redirect_url: str) -> str:
    """
    Follows a Piazza redirect URL without loading the body, returning the final CDN URL.
    """
    # Issue a HEAD-like GET that does not follow redirects
    resp = requests.get(redirect_url, allow_redirects=False)
    # If Piazza responded with a redirect, grab the Location header
    if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
        return resp.headers.get('Location')
    # Otherwise ensure no HTTP error
    resp.raise_for_status()
    # Fallback to original URL if no redirect
    return redirect_url

#fetch image data
piazza_url = "https://cdn-uploads.piazza.com/paste/lm7ul7s687n4fe/704802f944a73744684c85e0f9aef3f2b076ef89cf2947d4a9201d237824f09e/image.png"
image_url = to_cdn_url(piazza_url)
image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

#pass to llm
llm = ChatOpenAI(model_name="gpt-4o-mini")

message = {
    "role": "user",
    "content": [
        {
            "type": "text",
            "text": "Describe this image to someone who is struggling in the course. Describe all drawings and transcribe any text. Use up to 200 words",
        },
        {
            "type": "image",
            "source_type": "base64",
            "data": image_data,
            "mime_type": "image/png",
        },
    ],
}

response = llm.invoke([message])
print(response.text())