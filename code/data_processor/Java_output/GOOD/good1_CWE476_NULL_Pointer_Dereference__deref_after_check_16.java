    private void good1() throws Throwable
    {
        while(true)
        {
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
