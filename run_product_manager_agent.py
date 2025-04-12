import asyncio

from app.agent.product_manager_agent import ProductManagerAgent # Import the specialized agent
from app.logger import logger


async def main():
    agent = ProductManagerAgent() # Instantiate the specialized agent
    try:
        prompt = input("Ask a Product Management question: ")
        if not prompt.strip():
            logger.warning("Empty prompt provided.")
            return

        logger.info("Processing your request with ProductManagerAgent...")
        response = await agent.run(prompt)
        logger.info("ProductManagerAgent processing completed.")
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
