    public void bad() throws Throwable
    {
        switch (7)
        {
        case 7:
        {
            /* FLAW: Check for null, but still dereference the object */
            String myString = null;
            if (myString == null)
            {
                IO.writeLine(myString.length());
            }
        }
        break;
        default:
            /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
            IO.writeLine("Benign, fixed string");
            break;
        }
    }
