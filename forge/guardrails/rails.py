"""NeMo Guardrails integration -- library mode.

Provides input/output/retrieval rails for the NeuralForge pipeline,
with graceful degradation when nemoguardrails is not installed.
"""
import logging

import forge.config as config

logger = logging.getLogger(__name__)

# NeMo Guardrails may not be installed in dev/test environments
try:
    from nemoguardrails import RailsConfig, LLMRails  # type: ignore[import-untyped]

    HAS_GUARDRAILS = True
except ImportError:
    HAS_GUARDRAILS = False
    logger.warning("NeMo Guardrails not installed -- running without safety rails")


class GuardrailsEngine:
    """Wrapper around NeMo Guardrails with passthrough fallback.

    When *nemoguardrails* is not installed the engine is disabled and all
    checks return ``allowed=True`` with the original content unchanged.

    Args:
        config_dir: Path to the NeMo Guardrails config directory.
                    Defaults to :data:`forge.config.GUARDRAILS_CONFIG_DIR`.
    """

    def __init__(self, config_dir: str | None = None):
        self.enabled = HAS_GUARDRAILS and config.GUARDRAILS_ENABLED
        self._rails = None
        self._config_dir = config_dir or config.GUARDRAILS_CONFIG_DIR

        if self.enabled:
            try:
                rails_config = RailsConfig.from_path(self._config_dir)
                self._rails = LLMRails(rails_config)
            except Exception as exc:
                logger.error("Guardrails init failed: %s", exc)
                self.enabled = False

    # ------------------------------------------------------------------
    # Input rails
    # ------------------------------------------------------------------

    async def check_input(self, query: str) -> dict:
        """Run input rails (PII scrub, jailbreak detection, topic check).

        Returns
        -------
        dict
            ``{allowed: bool, reason: str | None, scrubbed_query: str}``
        """
        if not self.enabled or self._rails is None:
            return {"allowed": True, "reason": None, "scrubbed_query": query}

        try:
            result = await self._rails.generate_async(
                messages=[{"role": "user", "content": query}]
            )
            blocked = result.get("blocked", False)
            reason = result.get("reason") if blocked else None
            scrubbed = result.get("content", query) if not blocked else query
            return {
                "allowed": not blocked,
                "reason": reason,
                "scrubbed_query": scrubbed,
            }
        except Exception as exc:
            logger.error("Input rail check failed: %s", exc)
            return {"allowed": True, "reason": None, "scrubbed_query": query}

    # ------------------------------------------------------------------
    # Output rails
    # ------------------------------------------------------------------

    async def check_output(
        self, query: str, response: str, context: dict | None = None
    ) -> dict:
        """Run output rails (hallucination, attribution, provenance).

        Returns
        -------
        dict
            ``{allowed: bool, response: str, provenance: dict}``
        """
        if not self.enabled or self._rails is None:
            return {"allowed": True, "response": response, "provenance": {}}

        try:
            messages = [
                {"role": "user", "content": query},
                {"role": "assistant", "content": response},
            ]
            result = await self._rails.generate_async(messages=messages)
            blocked = result.get("blocked", False)
            final_response = result.get("content", response) if not blocked else response
            provenance = result.get("provenance", {})
            return {
                "allowed": not blocked,
                "response": final_response,
                "provenance": provenance,
            }
        except Exception as exc:
            logger.error("Output rail check failed: %s", exc)
            return {"allowed": True, "response": response, "provenance": {}}

    # ------------------------------------------------------------------
    # Full guarded pipeline
    # ------------------------------------------------------------------

    async def guarded_generate(
        self,
        query: str,
        context: dict | None = None,
        generate_fn=None,
    ) -> dict:
        """Full guarded generation: input rails -> generate -> output rails.

        Parameters
        ----------
        query:
            The user query.
        context:
            Optional retrieval context dict.
        generate_fn:
            Async callable ``(query, context) -> str`` for the actual LLM
            generation.  When *None*, returns the query echo (useful for
            testing the rail pipeline in isolation).

        Returns
        -------
        dict
            ``{response: str, input_check: dict, output_check: dict}``
        """
        # 1. Input rails
        input_check = await self.check_input(query)
        if not input_check["allowed"]:
            return {
                "response": f"[BLOCKED] {input_check['reason']}",
                "input_check": input_check,
                "output_check": {},
            }

        effective_query = input_check["scrubbed_query"]

        # 2. Generate
        if generate_fn is not None:
            response = await generate_fn(effective_query, context or {})
        else:
            response = effective_query  # echo for testing

        # 3. Output rails
        output_check = await self.check_output(effective_query, response, context)

        return {
            "response": output_check["response"],
            "input_check": input_check,
            "output_check": output_check,
        }
