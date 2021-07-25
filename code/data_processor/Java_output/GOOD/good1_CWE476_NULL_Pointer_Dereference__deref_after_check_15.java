    private void good1() throws Throwable
    {
        switch (8)
        {
        case 7:
            /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
            IO.writeLine("Benign, fixed string");
            break;
        default:
        {
            /* FIX: Check for null and do not dereference the object if it is null */
            String myString = null;
            if (myString == null)
            {
                IO.writeLine("The string is null");
            }
        }
        break;
        }
    }
