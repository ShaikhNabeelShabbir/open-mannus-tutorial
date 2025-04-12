import asyncio

from app.agent.data_eng_agent import DataEngAgent # Import the specialized agent
from app.logger import logger


async def main():
    agent = DataEngAgent() # Instantiate the specialized agent
    try:
        prompt = input("Ask a Data Engineering question: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.info("Processing your request with DataEngAgent...")
        response = await agent.run(prompt)
        logger.info("DataEngAgent processing completed.")
        print(f"\nAgent Response:\n{response}") # Print the final response

    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # Ensure agent resources are cleaned up
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
