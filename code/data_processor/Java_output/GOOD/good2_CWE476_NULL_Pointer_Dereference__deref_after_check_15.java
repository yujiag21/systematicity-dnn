    private void good2() throws Throwable
    {
        switch (7)
        {
        case 7:
        {
            /* FIX: Check for null and do not dereference the object if it is null */
            String myString = null;
            if (myString == null)
            {
                IO.writeLine("The string is null");
            }
        }
        break;
        default:
            /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
            IO.writeLine("Benign, fixed string");
            break;
        }
    }
