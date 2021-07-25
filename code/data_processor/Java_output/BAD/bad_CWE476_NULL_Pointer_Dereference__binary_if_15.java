    public void bad() throws Throwable
    {
        switch (7)
        {
        case 7:
        {
            String myString = null;
            /* FLAW: Using a single & in the if statement will cause both sides of the expression to be evaluated
             * thus causing a NPD */
            if ((myString != null) & (myString.length() > 0))
            {
                IO.writeLine("The string length is greater than 0");
            }
        }
        break;
        default:
            /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
            IO.writeLine("Benign, fixed string");
            break;
        }
    }
