    private void goodG2B1() throws Throwable
    {
        StringBuilder data;
        if (false)
        {
            /* INCIDENTAL: CWE 561 Dead Code, the code below will never run
             * but ensure data is inititialized before the Sink to avoid compiler errors */
            data = null;
        }
        else
        {

            /* FIX: hardcode data to non-null */
            data = new StringBuilder();

        }

        if (true)
        {
            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.length());
        }
    }
