import asyncio

from app.agent.marketing_lead_agent import MarketingLeadAgent
from app.logger import logger


async def main():
    agent = MarketingLeadAgent()
    try:
        prompt = input("Ask a Marketing question: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.info("Processing your request with MarketingLeadAgent...")
        response = await agent.run(prompt)
        logger.info("MarketingLeadAgent processing completed.")
        print(f"\nAgent Response:\n{response}")

    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # Ensure agent resources are cleaned up
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
