import asyncio
import httpx


async def ping_openclaw(base_url="http://127.0.0.1:18789"):
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(base_url)
            return {
                "ok": True,
                "status_code": r.status_code,
                "text_preview": r.text[:200],
            }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
        }


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    result = loop.run_until_complete(ping_openclaw())
    print(result)