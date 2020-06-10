dataset = ["Test-5", "Test-10", "Test-20", "Test-40"]
files = ["curve.5", "curve.10", "curve.20", "curve.40"]

#f = open("zwj-large.csv")
def readfile (files, dataset):
    f = open("zwj-random-large.csv", "w")
    f.write("Members,Error,Data\n")
    for i in range(len(files)):
        ff = open(files[i], "r")
        lines = ff.readlines()
        ff.close()
        count = 1
        for line in lines:
            error = 1 - float(line.split()[-1])
            f.write(str(count) + "," + str(error) + "," + dataset[i] + "\n")
            count += 1
            if count > 10:
                break
    f.close()

readfile(files, dataset)

