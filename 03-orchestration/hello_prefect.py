from prefect import flow, task, get_run_logger

@task
def say_hello():
    logger = get_run_logger()
    logger.info("Hello, World!")

@flow
def hello_world_flow():
    say_hello()

if __name__ == "__main__":
    hello_world_flow()
