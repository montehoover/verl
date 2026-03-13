
print("Total Bill (Rent + Utilities) SO FAR (no utilities for march): ")
aug = 750 + 74.43 + 4.83
sept = 750 + 64.79 + 79.34 + 6.08
oct = 750 + 64.79 + 79.34 + 6.08
nov = 750
dec = 750 + 25.07 + 100.68
jan = 750 + 32.99 + 41.00
feb = 750 + 78.70 + 39.40
mar = 750

total = aug + sept + oct + nov + dec + jan + feb + mar
print(f"Total Andrew owes: {total}")

payments = 750 + 750 + 79.26 + 1000 + 1649.30
print(f"Total Andrew has payed: {payments}")
print(f"Total Andrew owes: {total - payments}")