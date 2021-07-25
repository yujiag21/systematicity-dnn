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
                String myString = null;
                myString = "Hello";

                IO.writeLine(myString.length());

                /* FIX: Don't check for null since we wouldn't reach this line if the object was null */
                myString = "my, how I've changed";

                IO.writeLine(myString.length());
            }

        }
    }
