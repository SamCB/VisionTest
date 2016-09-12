from __future__ import division

from utils import from_dictionary

DISTANCE_THRESHOLD = 100


def compare_results_to_annotation(results, annotations):
    """Return a dictionary describing the 'correctness' of each class for
    results in a given annotation

    Return:
    A dictionary with class names as key and a dictionary as value.
        This sub dictionary contains three keys:
            "true_positives" for a tuple:
                (annotated_point,
                 (square_distance_from_annotated_point, result_point))
            "false_positives" for a result:
                result_point_that_wasn't_used
            "false_negatives" for an annotation:
                annotation_point_that_wasn't_assigned

    An example:
    {"Ball": {"true_positives": [({some_annotation}, (15.4, {some_result}))],
              "false_positives": [{some_other_result}],
              "false_negatives": []},
     "Corner": {"true_positives": [],
                "false_positives": [],
                "false_negatives": [{some_annotation},
                                    {some_other_annotation},
                                    {and_yet_another_annotation}]},
    ...
    }
    """
    annotated_classes = {}
    for class_, values in annotations:
        annotated_classes[class_] = (annotated_classes.get(class_, []) +
                                     [from_dictionary(values, name=class_)])
    result_classes = {}
    result_count_classes = {}
    for class_, values in results:
        result_classes[class_] = (result_classes.get(class_, []) +
                                  [from_dictionary(values, name=class_)])
        result_count_classes[class_] = (result_count_classes.get(class_, []) +
                                        [0])

    comparisons = {}
    for class_ in annotated_classes:
        true_positives = []
        false_negatives = []
        for a_idx, a in enumerate(annotated_classes[class_]):
            closest_result_idx = None
            closest_dist = float('inf')

            if class_ in result_classes:
                for r_idx, r in enumerate(result_classes[class_]):
                    diff = a.sq_difference(r)
                    if diff < closest_dist:
                        closest_result_idx = r_idx
                        closest_dist = diff

            if closest_dist > DISTANCE_THRESHOLD:
                # Problem of the DISTANCE_THRESHOLD is that it will be more
                #  lenient on points than rectangles and lines (more things to
                #  be off on a rectangle)
                # TODO: Determine if this is a problem.
                false_negatives.append(a)
            else:
                true_positives.append(
                    (a, (closest_dist, result_classes[class_][closest_result_idx]))
                )
                result_count_classes[class_][closest_result_idx] += 1

        comparisons[class_] = {"true_positives": true_positives,
                               "false_negatives": false_negatives,
                               "false_positives": []}

    for class_, results_counts in result_count_classes.items():
        false_positives = []
        for idx, result_count in enumerate(results_counts):
            if result_count == 0:
                false_positives.append(result_classes[class_][idx])
            elif result_count > 1:
                # This probably isn't good, but don't know how we want to
                #  record it right now
                pass

        if class_ not in comparisons:
            comparisons[class_] = {"true_positives": [],
                                   "false_negatives": []}

        comparisons[class_]["false_positives"] = false_positives

    return comparisons


def summarise_comparison(comparison):
    summary = {}
    for class_, class_comparison in comparison.items():
        summary[class_] = {
            'true_positives': len(class_comparison['true_positives']),
            'false_positives': len(class_comparison['false_positives']),
            'false_negatives': len(class_comparison['false_negatives'])
        }
        summary[class_]['class_size'] = sum(summary[class_].values())
        if class_comparison['true_positives']:
            summary[class_]['avg_true_positive_distance'] = sum(class_comparison['true_positives'][1][0]) / len(class_comparison['true_positives'])
        else:
            summary[class_]['avg_true_positive_distance'] = float('nan')

    return summary


def comparison_string(comparison=None, comparison_summary=None):
    if comparison_summary is None:
        comparison_summary = summarise_comparison(comparison)

    lines = []
    for class_, summary in comparison_summary.items():
        class_size = summary['class_size']
        num_true_positives = summary['true_positives']
        num_false_positives = summary['false_positives']
        num_false_negatives = summary['false_negatives']
        avg_tp_distance = summary['avg_true_positive_distance']

        lines.append("{}:".format(class_))
        lines.append(
            "    True Positives:  {}/{} = {:.2f} (avg dist: {:.2f})"
            .format(num_true_positives, class_size,
                    num_true_positives/class_size, avg_tp_distance))
        lines.append(
            "    False Positives: {}/{} = {:.2f}"
            .format(num_false_positives, class_size,
                    num_false_positives/class_size))
        lines.append(
            "    False Negatives: {}/{} = {:.2f}"
            .format(num_false_negatives, class_size,
                    num_false_negatives/class_size))

    return "\n".join(lines)
