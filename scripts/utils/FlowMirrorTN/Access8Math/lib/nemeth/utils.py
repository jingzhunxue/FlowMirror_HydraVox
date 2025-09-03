from collections import defaultdict
import csv


def nemeth2symbol_with_priority(src):
	with open(src, "r", encoding="utf8") as src_file:
		src_dict_reader = csv.DictReader(src_file)
		data = defaultdict(list)

		for row in src_dict_reader:
			if row["nemeth"] == "":
				continue
			data[row["nemeth"]].append({
				"latex": row["latex"],
				"symbol": row["symbol"],
				"priority": int(row["priority"]),
			})
			if "ingore-braille-space" in row and row["ingore-braille-space"] == "True":
				data[row["nemeth"].strip("⠀")].append({
					"latex": row["latex"],
					"symbol": row["symbol"],
					"priority": int(row["priority"]) + 100,
				})

	nemeth2symbol = {}
	for key, value in data.items():
		if len(value) > 1:
			temp = sorted(value, key=lambda i: i["priority"])
		else:
			temp = value
		if temp[0]["latex"].startswith("\\"):
			symbol = temp[0]["latex"] + " "
		else:
			symbol = temp[0]["symbol"]
		nemeth2symbol[key] = symbol

	return nemeth2symbol


if __name__ == "__main__":
	i = nemeth2symbol_with_priority("final.csv")
