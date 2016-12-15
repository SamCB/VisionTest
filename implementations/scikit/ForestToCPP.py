import numpy as np

from sklearn import tree, ensemble
from sklearn.tree import _tree

def convertForest(forest):
    """Converts a random forest to a cpp function.
    
    Parameters
    ----------
    forest: ensemble.RandomForestClassifier
        The forest from which a cpp function should be built.
    """

    # The number of histogram buckets the tree expects in inputs.
    histogramBuckets = 16

    # The set of leaf outputs.
    leafOuts = ""
    leafVals = []

    # The cpp file as a string.
    cppFile = """
#include "RandomForest.hpp"
#include <vector>

int RandomForest::classify(std::vector<float> histogram)
{
    int classifications;
"""

    # List of input labels.
    labels = ["histogram[" + str(i) + "]" for i in range(histogramBuckets)]

    # Run through each tree.
    for i, tree in enumerate(forest.estimators_):
        
        cppFile += "    const float* tree" + str(i) + ";\n"
        cppFile += treeToCode(tree, labels, "tree" + str(i), leafVals)
        cppFile += "\n"
    
    # Combine the output into a classification.
    cppFile += "    float classProbs[{}];\n".format(forest.n_classes_)
    for cls in range(forest.n_classes_):
        cppFile += "    classProbs[{}] = ".format(cls)
        for i, _ in enumerate(forest.estimators_):
            cppFile += "tree{}[{}] + ".format(i, cls)
        cppFile = cppFile[:-3] + ";\n"

    cppFile += """
    int classification = 0;
    float highestProb = 0.0;
"""
    for cls in range(forest.n_classes_):
        cppFile += """
    if(classProbs[{0}] > highestProb){{
        classification = {0};
        highestProb = classProbs[{0}];
    }}
""".format(cls)

    # Make the leaf outputs.
    for leaf, vals in enumerate(leafVals):
        leafOuts += "const float leafVal{}[{}] = {{{}}};\n".format(leaf, len(vals), str(vals)[1:-1])

    # Return and close.
    cppFile += "\n    return(classification);\n";
    cppFile += "}"

    with open("./implementations/scikit/CPPForest/cppForest.cpp", "w") as saveFile:
        saveFile.write(leafOuts + "\n" + cppFile)
        
# Converted from http://stackoverflow.com/questions/20224526/how-to-extract-the-decision-rules-from-scikit-learn-decision-tree
def treeToCode(tree, feature_names, treeVar, leafVals):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    def recurse(node, depth, leafVals):
        func = ""
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            func += "{}if({} <= {}){{\n".format(indent, name, threshold)
            func += recurse(tree_.children_left[node], depth + 1, leafVals)
            func += "{}}}\n".format(indent)
            func += "{}else {{  // if({} > {})\n".format(indent, name, threshold)
            func += recurse(tree_.children_right[node], depth + 1, leafVals)
            func += "{}}}\n".format(indent)
        else:
            # Convert what I think is something like counts (issue is they are floats) to probabilities.
            func += "{}{} = {};\n".format(indent, treeVar, "leafVal" + str(len(leafVals)))
            result = tree_.value[node][0]/sum(tree_.value[node][0])
            leafVals.append(result.tolist())
        return func

    return recurse(0, 1, leafVals)












































