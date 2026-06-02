import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger("ObsidianMCP")

class ObsidianVaultMCP:
    """Model Context Protocol (MCP) Server for JuniorCoach-Vault context ingestion."""

    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        if not self.vault_path.exists():
            logger.warning(f"Vault path {self.vault_path} does not exist. Creating local sandbox vault.")
            self.vault_path.mkdir(parents=True, exist_ok=True)

    def _scan_vault(self) -> Dict[str, str]:
        """Maps all markdown files in the vault."""
        index = {}
        for root, _, files in os.walk(self.vault_path):
            for file in files:
                if file.endswith(".md"):
                    full_path = Path(root) / file
                    relative_name = str(full_path.relative_to(self.vault_path))
                    index[relative_name] = str(full_path)
        return index

    async def handle_request(self, payload: Dict[str, Any]) -> str:
        """Processes JSON-RPC MCP requests."""
        method = payload.get("method")
        params = payload.get("params", {})

        if method == "list_tools":
            return json.dumps({
                "jsonrpc": "2.0",
                "id": payload.get("id"),
                "result": {
                    "tools": [{
                        "name": "read_vault_note",
                        "description": "Reads a specific markdown note from the JuniorCoach Obsidian Vault.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"note_name": {"type": "string", "description": "The exact name of the .md file."}},
                            "required": ["note_name"]
                        }
                    }, {
                        "name": "search_vault",
                        "description": "Returns a list of all available notes in the vault.",
                        "inputSchema": {"type": "object", "properties": {}}
                    }]
                }
            })

        elif method == "call_tool":
            tool_name = params.get("name")
            args = params.get("arguments", {})

            if tool_name == "search_vault":
                files = list(self._scan_vault().keys())
                content = "\n".join(files) if files else "Vault is empty."
                return json.dumps({"jsonrpc": "2.0", "id": payload.get("id"), "result": {"content": [{"type": "text", "text": content}]}})

            elif tool_name == "read_vault_note":
                note_name = args.get("note_name")
                index = self._scan_vault()
                if note_name in index:
                    with open(index[note_name], 'r', encoding='utf-8') as f:
                        text = f.read()
                    return json.dumps({"jsonrpc": "2.0", "id": payload.get("id"), "result": {"content": [{"type": "text", "text": text}]}})
                return json.dumps({"jsonrpc": "2.0", "id": payload.get("id"), "error": {"code": -32602, "message": "Note not found"}})

        return json.dumps({"jsonrpc": "2.0", "id": payload.get("id"), "error": {"code": -32601, "message": "Method not found"}})

    async def run_stdio(self):
        """Standard IO event loop for Cursor/Claude desktop app integration."""
        logger.info("Obsidian MCP Server listening on STDIO...")
        import sys
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, sys.stdin.readline)
            if not line:
                break
            try:
                payload = json.loads(line.strip())
                response = await self.handle_request(payload)
                sys.stdout.write(response + "\n")
                sys.stdout.flush()
            except json.JSONDecodeError:
                continue
