    public void bad() throws Throwable
    {
        if (true)
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
