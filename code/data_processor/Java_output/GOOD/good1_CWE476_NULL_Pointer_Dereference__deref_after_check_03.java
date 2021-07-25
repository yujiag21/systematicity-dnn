    private void good1() throws Throwable
    {
        if (5 != 5)
        {
            /* INCIDENTAL: CWE 561 Dead Code, the code below will never run */
            IO.writeLine("Benign, fixed string");
        }
        else
        {

            {
                /* FIX: Check for null and do not dereference the object if it is null */
                String myString = null;

                if (myString == null)
                {
                    IO.writeLine("The string is null");
                }
            }

        }
    }
