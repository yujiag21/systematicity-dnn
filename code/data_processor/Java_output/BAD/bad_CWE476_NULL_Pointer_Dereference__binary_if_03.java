    public void bad() throws Throwable
    {
        if (5 == 5)
        {
            {
                String myString = null;
                /* FLAW: Using a single & in the if statement will cause both sides of the expression to be evaluated
                 * thus causing a NPD */
                if ((myString != null) & (myString.length() > 0))
                {
                    IO.writeLine("The string length is greater than 0");
                }
            }
        }
    }
