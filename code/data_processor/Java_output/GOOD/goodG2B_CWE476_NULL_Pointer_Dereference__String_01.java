    private void goodG2B() throws Throwable
    {
        String data;

        /* FIX: hardcode data to non-null */
        data = "This is not null";

        /* POTENTIAL FLAW: null dereference will occur if data is null */
        IO.writeLine("" + data.length());

    }
