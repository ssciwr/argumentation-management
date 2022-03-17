import annotator.msomajo as msomajo
import annotator.base as be

text = "Das Feld der Computerchemie erfreut sich seit einiger Zeit einer stetig zunehmenden Aufmerksamkeit. Hierbei ist es für die Berechnung von chemisch relevanten Systemen von Interesse, möglichst große Systeme (Moleküle) durch effiziente Nutzung von Rechen- und Speicherressourcen zugänglich zu machen. Die Herausforderung besteht darin, dass mit der zunehmenden Anzahl von Atomen in größeren Systemen auch die Anzahl der internen Freiheitsgrade zunimmt."

if __name__ == "__main__":

    default_dict = be.prepare_run.load_input_dict("src/annotator/input")
    mydict = be.prepare_run.get_encoding(default_dict)

    msomajo.pretokenize(text, "de_CMC", mydict)
