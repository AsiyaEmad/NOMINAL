from typing import Annotated

from fastapi import APIRouter, Depends, Header
from fastapi.responses import JSONResponse

from backend.app.core.dependencies import get_chat_gateway
from backend.app.models import ChatCompletionRequest, OpenAIChatCompletionResponse
from backend.app.providers.errors import ProviderError
from backend.app.routing.gateway import ChatCompletionGateway

router = APIRouter()


@router.post(
    "/v1/chat/completions",
    response_model=OpenAIChatCompletionResponse,
    response_model_exclude_none=True,
)
async def create_chat_completion(
    request: ChatCompletionRequest,
    gateway: Annotated[ChatCompletionGateway, Depends(get_chat_gateway)],
    x_nominal_debug: Annotated[bool, Header()] = False,
) -> OpenAIChatCompletionResponse | JSONResponse:
    try:
        request_id, completion, decision = await gateway.complete(request)
    except ProviderError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "message": str(exc),
                    "type": exc.error_type,
                    "param": None,
                    "code": exc.code,
                }
            },
            headers={"x-request-id": exc.request_id} if exc.request_id else None,
        )

    content = OpenAIChatCompletionResponse.from_internal(completion).model_dump(exclude_none=True)
    if x_nominal_debug:
        content["nominal"] = {
            "request_id": request_id,
            "routing": decision.model_dump(mode="json"),
        }
    return JSONResponse(
        content=content,
        headers={"x-request-id": request_id},
    )
