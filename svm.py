import xlrd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn import metrics




def read_in_xlsx(file_name: str) -> type(list):
    """
    Reads in the first three columns of an excel sheet
    Input
        Filename and location
    Ouput
        x_data: the first column skipping the first element
        y_data: the second column skipping the first element
        label: the third column skipping the first element
    """
    # Open the document
    wb = xlrd.open_workbook(file_name)
    sheet = wb.sheet_by_index(0)
    sheet.cell_value(0, 0)

    # Read in the X data, Y data and labels and store them in a list of lists
    data = [[], [], []]
    for column in range(sheet.ncols):
        for row in range(sheet.nrows):
            if row == 0 :
                pass
            else:
                data[column].append(sheet.cell_value(row, column))

    dataset = []
    for x in range(len(data[0])):
        dataset.append([data[0][x], data[1][x]])

    # Return the X data, Y data and Labels
    return dataset, data[2]


def plot_dataset(dataset: list, labels: list, coefficients: list=[], intercept:list=[], support_vectors:list=[]) -> type(None):
    """
    Plot the dataset and the classification boundary as an optional
    Inout
        dataset: the data in [[x, y], [x, y]] format
        labels: the labes corresponding to the x, y pairs
        coefficients: the coefficients of the decision function
        intercept: the y coordinates of interception of the decision function
        support_vectors: optional argument to plot the support vectors used by the algorithm
    """
    # Arrays to store the sorted dataset
    class_1_x = []
    class_1_y = []
    class_2_x = []
    class_2_y = []

    # Sort the dataset based on classes
    for i in range(len(labels)):
        if labels[i] == 0:
            class_1_x.append(dataset[i][0])
            class_1_y.append(dataset[i][1])
        else:
            class_2_x.append(dataset[i][0])
            class_2_y.append(dataset[i][1])
    
    # Plot the dataset and the classification boundary
    fig, ax = plt.subplots()
    ax.scatter(class_1_x, class_1_y, label = "Class 1")
    ax.scatter(class_2_x, class_2_y, label = "Class 2")
    
    # If a boundary was provided plot it aswell
    if len(coefficients) >0:
        w = coefficients[0]
        # The slope of the function, the intersect is 
        a = -w[0]/w[1]

        # The x and y values
        result_x = range(5)
        result_y = []
        # Calculate the y values
        for x in result_x:
            result_y.append(a*x - (intercept[0]/w[1]))

        # Plot the function
        ax.plot(result_x, result_y, label = "Classification boundary")

    # If support vectors were given plot them too
    if len(support_vectors) > 0:
        x_support = []
        y_support = []
        for i in range(len(support_vectors)):
            x_support.append(support_vectors[i][0])
            y_support.append(support_vectors[i][1])

        ax.scatter(x_support, y_support,marker="x", label = "Support Vectors")

    leg = ax.legend()
    plt.show()



# Get the linearly separable dataset and train SVM on the dataset - 10 elements
dataset, labels = read_in_xlsx('Linear_SVM_Data.xls')
clf = svm.SVC(kernel='linear', C=1E10)
clf.fit(dataset[:-10], labels[:-10])
# Get the support vectors
support_vectors = clf.support_vectors_
# Get the coefficient of the decision function
coefficients = clf.coef_
# The interception
intercept = clf.intercept_

# Test our classifier on the last 10 elements of the dataset and print its accuracy
y_pred = clf.predict(dataset[-10:])
print("The accuracy of the classifier was:", metrics.accuracy_score(y_pred, labels[-10:]))
# Plot the classifier results and the dataset
plot_dataset(dataset, labels, coefficients=coefficients, intercept =intercept, support_vectors=support_vectors)
