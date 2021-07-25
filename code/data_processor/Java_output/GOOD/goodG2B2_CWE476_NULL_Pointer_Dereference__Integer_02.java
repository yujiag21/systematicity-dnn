    private void goodG2B2() throws Throwable
    {
        Integer data;
        if (true)
        {
            /* FIX: hardcode data to non-null */
            data = Integer.valueOf(5);
        }
        else
        {
            /* INCIDENTAL: CWE 561 Dead Code, the code below will never run
             * but ensure data is inititialized before the Sink to avoid compiler errors */
            data = null;
        }

        if (true)
        {
            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.toString());
        }
    }
