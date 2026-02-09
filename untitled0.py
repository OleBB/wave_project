#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 09:57:07 2026

@author: ole
"""

print("\n--- Python List Cheat Sheet (Runnable) ---\n")

# ----------------------------------------
# 1. append: adds ONE item
# ----------------------------------------
lst = []
print("Start lst:", lst)
# %%


lst.append("path")
print("After append('path'):", lst)
# %%


# append returns None
result = lst.append("more")
print("Return value of append:", result)
print("lst after append again:", lst)
# %%



# ----------------------------------------
# 2. extend: adds MANY items
# ----------------------------------------
lst = []
print("\nReset lst:", lst)
# %%


lst.extend(["a", "b", "c"])
print("After extend(['a','b','c']):", lst)
# %%


# extend with string (common gotcha)
lst = []
lst.extend("path")
print("After extend('path'):", lst)
# %%



# ----------------------------------------
# 3. append vs extend
# ----------------------------------------
lst = []
lst.append(["a", "b"])
print("\nappend(['a','b']):", lst)
# %%


lst = []
lst.extend(["a", "b"])
print("extend(['a','b']):", lst)

# %%


# ----------------------------------------
# 4. Variables point to the SAME list
# ----------------------------------------
original = ["x", "y"]
alias = original

print("\noriginal:", original)
print("alias:", alias)
# %%


alias.append("z")
print("After alias.append('z'):")
print("original:", original)
print("alias:", alias)

# %%


# ----------------------------------------
# 5. Reassigning does NOT reset original
# ----------------------------------------
alias = []
print("\nAfter alias = []")
print("original:", original)
print("alias:", alias)

# %%


# ----------------------------------------
# 6. Copying a list (SAFE)
# ----------------------------------------
safe_copy = original.copy()
safe_copy.append("SAFE")

print("\nAfter copying and modifying copy:")
print("original:", original)
print("safe_copy:", safe_copy)
# %%



# ----------------------------------------
# 7. Object identity proof
# ----------------------------------------
print("\nObject IDs:")
print("id(original): ", id(original))
print("id(safe_copy):", id(safe_copy))


print("\n--- End of Cheat Sheet ---\n")
