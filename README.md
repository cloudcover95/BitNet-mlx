# BitNet-MLX: Sovereign MCP Vanguard

**Production-grade MLX-native ternary (1.58-bit) SDK and MCP Server for Apple Silicon.**

## Model Sovereignty via MCP
BitNet-MLX executes as an autonomous edge-native MCP server. AI code assistants (Cursor, Claude Desktop) can locally query real-time value compute capabilities (UMA RAM, thermal load) and execute parametric data audits directly on your edge hardware.

### MCP Gateway Configuration
```json
{
  "mcpServers": {
    "bitnet-mlx-sovereign": {
      "command": "bitnet-mlx",
      "args": ["mcp"]
    }
  }
}
System Capabilities
Zero-SVD Operations: Strict O(N) execution mapped natively to AMX registers.

Dynamic Kinematic Routing: Automatic degradation based on 48V power limits or system thermal constraints.

Parametric Weighting: Grok data inference tracking saved immutably to .parquet format for agent retrieval.
