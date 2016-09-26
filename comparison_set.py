from __future__ import division
from math import isnan

from comparison import summarise_comparison

class ComparisonSet:

    def __init__(self):
        self.comparisons = []
        self.get_all_comparisons_results = None
        self.get_averages_results = None
        self.get_overall_averages_results = None

    def add(self, comparison):
        self.comparisons.append(comparison)
        self.get_all_comparisons_results = None
        self.get_averages_results = None
        self.get_overall_averages_results = None

    def get_all_comparisons(self):
        if self.get_all_comparisons_results:
            return self.get_all_comparisons_results

        classes = {}
        for comparison in self.comparisons:
            summary = summarise_comparison(comparison)
            for class_, class_summary in summary.items():
                if class_ not in classes:
                    classes[class_] = {
                        'true_positives': [],
                        'false_positives': [],
                        'false_negatives': [],
                        'avg_true_positive_distance': []
                    }
                for metric, result in class_summary.items():
                    if(metric != 'avg_true_positive_distance' and
                       not isnan(result)):
                        try:
                            classes[class_][metric].append(result)
                        except KeyError:
                            pass

        self.get_all_comparisons_results = classes
        return classes

    def get_averages(self):
        if self.get_averages_results:
            return self.get_averages_results

        comparisons = self.get_all_comparisons()
        averages = {}
        for class_, class_comparisons in comparisons.items():
            averages[class_] = {}
            for metric, results in class_comparisons.items():
                try:
                    averages[class_][metric] = sum(results)/len(results)
                except ZeroDivisionError:
                    averages[class_][metric] = float('nan')

        self.get_averages_results = averages
        return averages

    def get_overall_averages(self):
        if self.get_overall_averages_results:
            return self.get_overall_averages_results
        all_class_averages = self.get_averages()
        averages = {}
        for class_averages in all_class_averages.values():
            for metric, results in class_averages.items():
                if metric not in averages:
                    averages[metric] = results
                else:
                    averages[metric] += results

        overall_averages = {
            metric: total/len(all_class_averages)
            for metric, total in averages.items()
        }
        self.get_overall_averages_results = overall_averages
        return overall_averages

    def print_averages(self):
        averages = self.get_averages()
        classes = [class_ for class_ in averages]
        lines = {
            "heading": " " * 16 + " | " + " | ".join("{:>12.12}".format(class_) for class_ in classes) + " |",
            "horizontal_line": "-" * 16 + "-=-" + "-=-".join("-" * 12 for _ in classes) + "-="
        }
        for class_ in classes:
            metrics = averages[class_]
            for metric, results in metrics.items():
                if metric not in lines:
                    lines[metric] = "{:>16.16} | ".format(metric)
                lines[metric] += "{:12.2f} | ".format(results)

        print
        print "Averages:"
        print
        print lines.pop("heading")
        print lines.pop("horizontal_line")
        print "\n".join(l for l in lines.values())
        print

        overall_averages = self.get_overall_averages()

        print "Overall Averages:"
        print
        print "\n".join("{:>16.16} | {:12.2f} |".format(metric, result) for metric, result in overall_averages.items())
