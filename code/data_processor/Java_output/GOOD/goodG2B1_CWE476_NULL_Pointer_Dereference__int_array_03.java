    private void goodG2B1() throws Throwable
    {
        int [] data;
        if (5!=5)
        {
            /* INCIDENTAL: CWE 561 Dead Code, the code below will never run
             * but ensure data is inititialized before the Sink to avoid compiler errors */
            data = null;
        }
        else
        {

            /* FIX: hardcode data to non-null */
            data = new int[5];

        }

        if (5==5)
        {
            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.length);
        }
    }
