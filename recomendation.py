#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 20 09:24:24 2025

@author: muhammadumair
"""

# Example tumor-drug mapping
tumor_drug_mapping = {
    "Glioma": ["Temozolomide", "Carmustine", "Lomustine"],
    "Meningioma": ["Sunitinib", "Everolimus", "Bevacizumab"],
    "Pituitary": ["Cabergoline", "Bromocriptine", "Octreotide"],
    "No-Tumor": ["No drug required"]
}


def suggest_drugs(tumor_type):
    """
    Suggest drugs based on the classified tumor type.
    """
    drugs=tumor_drug_mapping.get(tumor_type, ["Unknown tumor type - no recommendations available"])
    return drugs


# Example usage



