    private void good2() throws Throwable
    {
        switch (7)
        {
        case 7:
        {
            String myString = null;
            myString = "Hello";
            IO.writeLine(myString.length());
            /* FIX: Don't check for null since we wouldn't reach this line if the object was null */
            myString = "my, how I've changed";
            IO.writeLine(myString.length());
        }
        break;
        default:
            /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
            IO.writeLine("Benign, fixed string");
            break;
        }
    }
