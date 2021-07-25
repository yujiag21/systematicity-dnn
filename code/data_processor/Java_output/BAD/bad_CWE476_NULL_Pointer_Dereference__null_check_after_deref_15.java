    public void bad() throws Throwable
    {
        switch (7)
        {
        case 7:
        {
            String myString = null;
            myString = "Hello";
            IO.writeLine(myString.length());
            /* FLAW: Check for null after dereferencing the object. This null check is unnecessary. */
            if (myString != null)
            {
                myString = "my, how I've changed";
            }
            IO.writeLine(myString.length());
        }
        break;
        default:
            /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
            IO.writeLine("Benign, fixed string");
            break;
        }
    }
