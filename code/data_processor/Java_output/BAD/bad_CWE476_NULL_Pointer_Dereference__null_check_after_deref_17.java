    public void bad() throws Throwable
    {
        for(int j = 0; j < 1; j++)
        {
            {
                String myString = null;
                myString = "Hello";
                IO.writeLine(myString.length());
                /* FLAW: Check for null after dereferencing the object. This null check is unnecessary. */
                if (myString != null)
                {
                    myString = "my, how I've changed";
                }
                IO.writeLine(myString.length());
            }
        }
    }
