import asyncio

from app.agent.hr_lead_agent import HRLeadAgent
from app.logger import logger


async def main():
    agent = HRLeadAgent()
    try:
        prompt = input("Ask a Human Resources question: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.info("Processing your request with HRLeadAgent...")
        response = await agent.run(prompt)
        logger.info("HRLeadAgent processing completed.")
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
