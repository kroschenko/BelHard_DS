from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


class Interactor:

    @staticmethod
    def choose_model() -> object:
        print("""Choose model:\n
            1. Naive Bayes
            2. Logistic Regression
            3. k-Nearest Neighbors
            """)

        while True:
            try:
                choice = int(input())
                if choice == 1:
                    model = GaussianNB()
                    break
                elif choice == 2:
                    while True:
                        try:
                            epochs = int(input("Enter number of epochs: "))
                            model = LogisticRegression(max_iter=epochs)
                            break
                        except ValueError:
                            print("Please type a number")
                    break
                elif choice == 3:
                    while True:
                        try:
                            k = int(input("Enter k value: "))
                            model = KNeighborsClassifier(n_neighbors=k)
                            break
                        except ValueError:
                            print("Please type a number")
                    break
                else:
                    print("Please enter a value from a proposed menu.")
            except ValueError:
                print("Please enter a valid choice.")

        return model

    @staticmethod
    def print_metrics(metrics: list) -> None:
        print(f"""
            1. Confussion Matrix:\n{metrics[0]}
            2. Accuracy: {metrics[1]}
            3. Precision: {metrics[2]}
            4. F1 Score: {metrics[3]}
            5. Recall: {metrics[4]}
            """)
