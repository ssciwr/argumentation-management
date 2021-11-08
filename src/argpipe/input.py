# the input class objects
import requests as rq
from bs4 import BeautifulSoup as BS
import pandas as pd

# parent class that includes the cleaning methods
class clean_input:
    """Parent class containing the clean-up methods.
    
    Args:
        inputStream (object): The input object.
        objectType (string): The type of input object.
        
    """

    def __init__(self, outdir, outfile):
        self.outdir = outdir
        self.outfile = outfile

    def request_URI(self, inputURI):
        """Requesting the resource using the provided URI."""
        resource = rq.get(inputURI, allow_redirects=True)
        # Check here that response is 200
        print(resource.status_code)
        f = open('test_aerzteblatt2.html', 'wb')
        f.write(resource.content)
        f.close
        return resource.content

    def clean_tags(self, resource):
        """Clean html file and remove xml tags."""
        soup = BS(resource, "html.parser")
        return soup.get_text()

    def get_article_body(self, resource):
        """Get only the article body from the html."""
        soup = BS(resource, "html.parser")
        mytext = [item.get_text() for item in soup.findAll('p')]
        return mytext

    def save_text(self, resource, langTag, fileTag):
        """Save the cleaned text in a text file."""
        # need to first check that directory is there, if not create it
        # also if files in there would be overwritten
        name = self.outdir + '/' + self.outfile + '-' + langTag + '-' + fileTag + '.txt'
        f = open(name, 'w', encoding='utf-8')
        f.write(str(resource))
        f.close

# a csv containing URIs 
class read_df(clean_input):
    """Class reading and grabbing the URI objects from
    an input csv; inherits methods from clean_input.
    
    Args:
        fileName (string): File name of the csv input file.
        
    """

    def __init__(self, outdir, outfile, fileName):
        self.fn = fileName
        # invoking the __init__ of the parent class 
        clean_input.__init__(self, outdir, outfile) 

    def read_csv(self):
        """Read csv file with URIs. Language one in column 2 and language two in column 3."""
        df = pd.read_csv(self.fn, r'\s+')
        lang1 = df.columns[1]
        lang2 = df.columns[2]
        print("Found languages {} and {}.".format(lang1,lang2))
        # need to check here that lists are complete and that there is a 1:1 correspondence of the 
        # two languages
        URIlang1 = df.iloc[:,1].tolist()
        URIlang2 = df.iloc[:,2].tolist()
        return lang1, lang2, URIlang1, URIlang2


def main(inputFile):
    """Main function call if analysis package is to be run as a script.

    Args:
        inputFile (string): Input file name.
    """
    obj = read_df('../test/', 'output', inputFile)
    lang1, lang2, URIlang1, URIlang2 = obj.read_csv()
    for i, item in enumerate(URIlang1):
        resource = obj.request_URI(item)
        # resource = obj.clean_tags(resource)
        resource = obj.get_article_body(resource)
        obj.save_text(resource, lang1,str(i+1))
    for i, item in enumerate(URIlang2):
        resource = obj.request_URI(item)
        # resource = obj.clean_tags(resource)
        resource = obj.get_article_body(resource)
        obj.save_text(resource, lang2,str(i+1))

if __name__ == "__main__":
    main('../test/aerzteblatt.csv')