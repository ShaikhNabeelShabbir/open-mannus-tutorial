import asyncio

from app.agent.finance_lead_agent import FinanceLeadAgent
from app.logger import logger


async def main():
    agent = FinanceLeadAgent()
    try:
        prompt = input("Ask a Finance or Financial Management question: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.info("Processing your request with FinanceLeadAgent...")
        response = await agent.run(prompt)
        logger.info("FinanceLeadAgent processing completed.")
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
