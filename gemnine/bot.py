"""
ChatGPT wrapper
"""

from __future__ import annotations
import httpx
from typing import Any, Self, AsyncGenerator, Literal
from typing import TypeAlias, cast, get_args, overload
from enum import Enum
from pydantic import BaseModel as PyBaseModel, Field, validate_call, PrivateAttr
from pydantic import model_validator, ConfigDict
from pydantic import field_serializer, model_serializer
# from PIL import Image
# from pathlib import Path
# from base64 import b64encode, b64decode
# from io import BytesIO
# from vermils.io import aio
from vermils.gadgets import mimics


DEFAULT_CONFIG = ConfigDict(
    validate_assignment=True,
    populate_by_name=True,
    extra="allow")

DEFAULT_HOST = "generativelanguage.googleapis.com"

SafetyBlockThreshold: TypeAlias = Literal[
    "BLOCK_NONE",
    "BLOCK_ONLY_HIGH",
    "BLOCK_MEDIUM_AND_ABOVE",
    "BLOCK_LOW_AND_ABOVE",
]

SafetyViolationProbability: TypeAlias = Literal[
    "NEGLIGIBLE",
    "LOW",
    "MEDIUM",
    "HIGH",
]

SafetyCategory: TypeAlias = Literal[
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
]


@mimics(PyBaseModel.model_dump)
def _dump(m: PyBaseModel, **kw):
    return m.model_dump(exclude_none=True, by_alias=True, mode="json")


class BaseModel(PyBaseModel):
    model_config = DEFAULT_CONFIG


class ModelInfo(BaseModel):
    name: str
    version: str
    display_name: str = Field(alias="displayName")
    description: str
    input_limit: int = Field(alias="inputTokenLimit")
    output_limit: int = Field(alias="outputTokenLimit")
    supported_methods: list[str] = Field(alias="supportedGenerationMethods")
    temperature: float | None = None
    top_p: float | None = Field(alias="topP", default=None)
    top_k: int | None = Field(alias="topK", default=None)


class Role(str, Enum):
    User = "user"
    Model = "model"
    Assistant = "model"
    """Alias for 'Model'"""
    Function = "function"


class Safety(BaseModel):
    category: SafetyCategory
    threshold: SafetyBlockThreshold | None = None
    probability: SafetyViolationProbability | None = None


class Message(BaseModel):

    class BaseSegment(BaseModel):
        model_config = ConfigDict(
            validate_assignment=True,
            populate_by_name=True,
            extra="allow",
            frozen=True,
        )

    class TextSegment(BaseSegment):
        text: str

        def __str__(self):
            return self.text

        def __eq__(self, other):
            if not isinstance(other, Message.TextSegment):
                return False
            return self.text == other.text

        def __hash__(self):
            return hash(self.text)

    class ImageSegment(BaseSegment):
        image_url: str
        image_format: str | None = None

        @model_serializer(when_used="json")
        def jsonable_serialize(self):
            return {
                "inlineData": {
                    "mimeType": f"image/{self.image_format or '*'}",
                    "data": self.image_url
                }
            }

        def __str__(self):
            return f"#image({self.image_url})"

        def __eq__(self, other):
            if not isinstance(other, Message.ImageSegment):
                return False
            return self.image_url == other.image_url

        def __hash__(self):
            return hash(self.image_url)

    class FuncCallSegment(BaseSegment):
        ...

    class FuncReturnSegment(BaseSegment):
        ...

    role: Role
    content: str | list[TextSegment | ImageSegment
                        | FuncCallSegment | FuncReturnSegment] = Field(alias="parts", default='')
    _tokens: int | None = None
    _hash: int | None = None

    @field_serializer("content", when_used="json")
    def convert_content(self, value, _) -> list[dict]:
        if isinstance(value, str):
            return [{"text": value}]
        return list(_dump(c) for c in value)

    @model_validator(mode="after")
    def post_init(self) -> Self:
        content_hash = (
            hash(self.content)
            if isinstance(self.content, str)
            else hash(tuple(self.content))
        )
        if self._hash is None or self._hash != content_hash:
            self._tokens = None
            self._hash = content_hash

        return self

    def __str__(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return ''.join(str(c) for c in self.content)


SegTypes: TypeAlias = (Message.TextSegment | Message.ImageSegment
                       | Message.FuncCallSegment | Message.FuncReturnSegment)
Prompt: TypeAlias = str | list[SegTypes]


class FullResponse(BaseModel):
    class Candidate(BaseModel):
        content: Message
        finish_reason: str = Field(alias="finishReason")
        index: int
        safety_ratings: list[Safety] = Field(alias="safetyRatings")

    class PromptFeedback(BaseModel):
        safety_ratings: list[Safety] = Field(alias="safetyRatings")

    candidates: list[Candidate]
    prompt_feedback: PromptFeedback = Field(alias="promptFeedback")


class Bot(BaseModel):
    """
    # Gemini bot

    ## Parameters
    - `model` (str): Model to use
    - `api_key` (str): OpenAI API key
    - `prompt` (str): Initial prompt
    - `temperature` (float): The higher the temperature, the crazier the text (0.0 to 1.0)
    - `comp_tokens` (float | int): Reserved tokens for completion, when value is in (0, 1), it represents a ratio,
        [1,-] represents tokens count, 0 for auto mode
    - `top_p` (float): Nucleus sampling: limits the generated guesses to a cumulative probability. (0.0 to 1.0)
    - `proxies` (dict[str, str]): Connection proxies
    - `timeout` (float | None): Connection timeout
    """

    model: str
    api_key: str
    api_host: str = DEFAULT_HOST
    comp_tokens: float = 0.0

    # See https://ai.google.dev/docs/concepts#model_parameters
    max_reply_tokens: int | None = None
    stop: list[str] | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Literal["auto", "none"] | dict[str, Any] = "auto"
    safety_settings: list[Safety] = Field(
        alias="safetySettings", default_factory=lambda: [
            Safety(category=c, threshold="BLOCK_NONE")
            for c in get_args(SafetyCategory)])

    proxy: str | None = None
    timeout: float | None = None
    _cli: httpx.AsyncClient = PrivateAttr(default=None)
    _minfo: ModelInfo | None = None
    _last_proxy: str | None = None
    _last_timeout: float | None = None
    _last_model: str | None = None

    @model_validator(mode="after")
    def post_init(self) -> Self:
        if (cast(None | httpx.AsyncClient, self._cli) is None
                or self.proxy != self._last_proxy or self.timeout != self._last_timeout):
            self.respawn_cli()
        if self.model != self._last_model:
            self._minfo = None
        return self

    def respawn_cli(self, **kw):
        """
        Create a new HTTP client, replacing the old one if it exists.
        """
        self._cli = httpx.AsyncClient(proxy=self.proxy,
                                      timeout=self.timeout,
                                      trust_env=False,
                                      **kw)

    def new_session(self, **kw) -> Session:
        return Session(bot=self, **kw)

    def _get_json(
            self,
            session: Session,
    ) -> dict:
        ret: dict = {
            "contents": list(_dump(m) for m in session.messages),
            "generationConfig": {
                "stopSequences": self.stop,
                "maxOutputTokens": self.max_reply_tokens,
                "topP": self.top_p,
                "topK": self.top_k,
                "temperature": self.temperature,
            }
        }
        if self.safety_settings:
            ret["safetySettings"] = list(_dump(s) for s in self.safety_settings)
        # print(f"json: {ret}")
        return ret

    async def get_model_info(
            self, name: str | None = None, refresh=False) -> ModelInfo:
        name = self.model if name is None else name
        if name == self.model and self._minfo is not None and not refresh:
            return self._minfo
        r = await self._cli.get(
            f"https://{self.api_host}/v1/{self.model}",
            headers={"x-goog-api-key": self.api_key})

        r.raise_for_status()

        m = ModelInfo.model_validate_json(r.text)
        if m.name == self.model:
            self._minfo = m
        return m

    async def list_models(self) -> list[ModelInfo]:
        r = await self._cli.get(
            f"https://{self.api_host}/v1/models",
            headers={"x-goog-api-key": self.api_key})

        r.raise_for_status()

        ms = list(ModelInfo.model_validate(m) for m in r.json()["models"])
        for m in ms:
            if m.name == self.model:
                self._minfo = m
                break
        return ms

    async def count_tokens(self, msgs: list[Message] | Message) -> int:
        if isinstance(msgs, Message):
            if msgs._tokens is not None:
                return msgs._tokens
            r = await self._cli.post(
                f"https://{self.api_host}/v1/{self.model}:countTokens",
                headers={"x-goog-api-key": self.api_key},
                json={"contents": [_dump(msgs)]})
            msgs._tokens = cast(int, r.json()["totalTokens"])
            return msgs._tokens

        if all(m._tokens is not None for m in msgs):
            return sum(cast(int, m._tokens) for m in msgs)

        r = await self._cli.post(
            f"https://{self.api_host}/v1/{self.model}:countTokens",
            headers={"x-goog-api-key": self.api_key},
            json={"contents": list(_dump(m) for m in msgs)})
        return r.json()["totalTokens"]

    async def stream(self,
                     prompt: Prompt,
                     role: Role = Role.User,
                     session: Session | None = None,
                     ) -> AsyncGenerator[str, None]:
        session = self.new_session() if session is None else session
        session.append(Message(role=role, content=prompt))
        await session.trim()

        async with self._cli.stream(
            "POST",
            f"https://{self.api_host}/v1/{self.model}:streamGenerateContent",
            headers={"x-goog-api-key": self.api_key},
            json=self._get_json(session)
        ) as r:
            if r.status_code != 200:
                session.pop()
                await r.aread()
                raise RuntimeError(
                    f"{r.status_code} {r.reason_phrase} {r.text}",
                )
            async for line in r.aiter_lines():
                line = line.strip()
                if line.startswith("\"text\": "):
                    line = line.removeprefix("\"text\": ")
                    line = line.strip('"').replace("\\n", '\n')
                    yield line

    async def send_raw(self,
                       prompt: Prompt,
                       role: Role = Role.User,
                       session: Session | None = None,
                       ) -> FullResponse:
        session = self.new_session() if session is None else session
        session.append(Message(role=role, content=prompt))
        await session.trim()

        r = await self._cli.post(
            f"https://{self.api_host}/v1/{self.model}:generateContent",
            headers={"x-goog-api-key": self.api_key},
            json=self._get_json(session)
        )

        r.raise_for_status()
        # print(f"\n\nres: {r.text}")
        return FullResponse.model_validate_json(r.text)

    async def send(self,
                   prompt: Prompt,
                   role: Role = Role.User,
                   session: Session | None = None,
                   ) -> str:
        r = await self.send_raw(prompt, role, session)
        return cast(Message.TextSegment,
                    r.candidates[0].content.content[0]).text.replace("\\n", '\n')


class Session(BaseModel):
    bot: Bot | None = Field(exclude=True)
    messages: list[Message] = Field(default_factory=list)

    async def trim(self, target_max: int | None = None, bot: Bot | None = None) -> int:
        """
        Trim the session until it's less than `target_max` tokens.
        Returns the total number of tokens after trimming.
        """
        bot = self.bot if bot is None else bot
        if bot is None:
            raise ValueError("bot is not set")
        if target_max is None:
            model_info = await bot.get_model_info()
            model_max_tokens = model_info.input_limit
            if bot.comp_tokens < 1:
                target_max = int((1-bot.comp_tokens) * model_max_tokens)
            else:
                target_max = max(0, model_max_tokens-int(bot.comp_tokens))

        msg_cnt = 0
        num_tokens = 0
        for message in reversed(self.messages):
            this_tokens = await bot.count_tokens(message)
            if num_tokens + this_tokens >= target_max:
                break
            msg_cnt += 1
            num_tokens += this_tokens
        self.messages = self.messages[len(self.messages)-msg_cnt:]
        return num_tokens

    @validate_call
    def append(self, msg: Message) -> None:
        self.messages.append(msg)

    def pop(self, index: int = -1) -> Message:
        return self.messages.pop(index)

    def rollback(self, num: int):
        """
        Roll back `num` messages.
        """
        self.messages = self.messages[:len(self.messages)-num]

    def clear(self):
        """
        Clear the session.
        """
        self.messages.clear()

    @validate_call
    async def stream(self,
                     prompt: Prompt,
                     role: Role = Role.User,
                     ) -> AsyncGenerator[str, None]:
        """
        Stream messages from the bot.

        ## Parameters
        - `prompt` (str): What to say
        - `role` (Role): Role of the speaker
        """
        content = ''
        if self.bot is None:
            raise ValueError("bot is not set")
        async for r in self.bot.stream(prompt, role, self):
            content += r
            yield r

        self.append(Message(role=Role.Model, content=content))

    @validate_call
    async def send(self,
                   prompt: Prompt,
                   role: Role = Role.User,
                   ) -> str:
        """
        Send a message to the bot.

        ## Parameters
        - `prompt` (str): What to say
        - `role` (Role): Role of the speaker
        """
        if self.bot is None:
            raise ValueError("bot is not set")
        ret = await self.bot.send(prompt, role, self)
        self.append(Message(role=Role.Model, content=ret))
        return ret

    def __iter__(self):
        return iter(self.messages)

    def __len__(self):
        return len(self.messages)

    def __bool__(self):
        return bool(self.messages)

    def __reversed__(self):
        return reversed(self.messages)

    @overload
    def __getitem__(self, index: int) -> Message:
        ...

    @overload
    def __getitem__(self, s: slice) -> list[Message]:
        ...

    def __getitem__(self, index: int | slice):
        return self.messages[index]

    @validate_call
    def __setitem__(self, index: int, value: Message):
        self.messages[index] = value
