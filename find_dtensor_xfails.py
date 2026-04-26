for r in open("log_ops_dtensor.out", "r"):
    r = r.strip()
    if r.startswith("ERROR: "):
        print(r.split("ERROR: ")[1].split(" ")[0])
