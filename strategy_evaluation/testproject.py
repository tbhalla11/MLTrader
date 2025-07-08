import experiment1 as exp1
import experiment2 as exp2


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "tbhalla6"  # replace tb34 with your Georgia Tech username.

if __name__ == "__main__":

    exp1.experiment1()
    exp2.experiment2()
    auth = author()
    print("Author is: " + auth)