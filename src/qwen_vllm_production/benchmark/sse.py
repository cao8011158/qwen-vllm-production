"""Incremental UTF-8 SSE framing, independent of HTTP byte boundaries."""

import codecs


class StreamProtocolError(ValueError):
    pass


class SSEDecoder:
    def __init__(self, max_event_chars: int = 1_048_576) -> None:
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.buffer = ""
        self.data: list[str] = []
        self.max_event_chars = max_event_chars

    def feed(self, chunk: bytes, *, final: bool = False) -> list[str]:
        self.buffer += self.decoder.decode(chunk, final=final)
        if final and self.buffer:
            self.buffer += "\n"
        events = []
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                if self.data:
                    events.append("\n".join(self.data))
                    self.data = []
            elif line.startswith("data:"):
                self.data.append(line[5:].removeprefix(" "))
            # Comments, event IDs and retry fields do not represent content.
            if len(self.buffer) + sum(map(len, self.data)) > self.max_event_chars:
                raise StreamProtocolError("SSE event exceeds the maximum allowed size.")
        if len(self.buffer) + sum(map(len, self.data)) > self.max_event_chars:
            raise StreamProtocolError("SSE event exceeds the maximum allowed size.")
        if final and self.data:
            events.append("\n".join(self.data))
            self.data = []
        return events
