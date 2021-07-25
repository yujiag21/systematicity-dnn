    public void bad() throws Throwable
    {
        for(int j = 0; j < 1; j++)
        {
            {
                /* FLAW: Check for null, but still dereference the object */
                String myString = null;
                if (myString == null)
                {
                    IO.writeLine(myString.length());
                }
            }
        }
    }
