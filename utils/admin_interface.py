def admin_approval(message):

    print("\n==============================")
    print("ADMIN APPROVAL REQUIRED")
    print(message)

    while True:

        decision = input("Approve? (y/n): ").lower()

        if decision == "y":
            return True

        if decision == "n":
            return False

        print("Enter y or n")