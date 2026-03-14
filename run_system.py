from master_agent.master_graph import master_graph


def run_system():

    print("\n===== CRISIS MANAGEMENT AI =====")

    while True:

        image = input("\nEnter satellite image path (or exit): ")

        if image == "exit":
            break

        master_graph.invoke({
            "satellite_image": image
        })


if __name__ == "__main__":
    run_system()