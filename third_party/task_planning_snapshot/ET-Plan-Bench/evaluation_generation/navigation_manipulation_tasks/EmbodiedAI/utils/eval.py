import re


def get_action_object_list(test_str):
    action = re.findall(r'\[.*?\]', test_str)
    objects = re.findall(r'\<.*?\>', test_str)

    # return action + objects[1:]

    action_result = action + objects[1:]

    action_string = ''
    for item in action_result:
        action_string = action_string + item[1:-1].lower() + " "

    return action_string


def lcs(X, Y):
    # Find the length of the lists
    m = len(X)
    n = len(Y)

    # Create a 2D array to store the lengths of LCS
    L = [[None]*(n+1) for i in range(m+1)]

    # Fill in the 2D array using dynamic programming approach
    for i in range(m+1):
        for j in range(n+1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i-1] == Y[j-1]:
                L[i][j] = L[i-1][j-1] + 1
            else:
                L[i][j] = max(L[i-1][j], L[i][j-1])

    # Collect the LCS from the 2D array
    index = L[m][n]
    lcs = [""] * (index+1)
    lcs[index] = ""

    i = m
    j = n
    while i > 0 and j > 0:
        if X[i-1] == Y[j-1]:
            lcs[index-1] = X[i-1]
            i -= 1
            j -= 1
            index -= 1
        elif L[i-1][j] > L[i][j-1]:
            i -= 1
        else:
            j -= 1

    return lcs
